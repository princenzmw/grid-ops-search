"""Bootstrap the grid-ops-search schema and load the operations-note corpus.

Usage:
    python -m src.init_db                    # load notes with their own timestamps
    python -m src.init_db --anchor-to-data   # remap note timestamps into the
                                             # available telemetry window so the
                                             # hybrid join demo returns rows

Idempotent: re-running refreshes embeddings and timestamps without
duplicating rows (upsert on note_id).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

import psycopg

from src.config import PROJECT_ROOT, SUBSTATION_ID_MAP, load_settings
from src.db import connect, log_connect, resolve_substation_map, substation_ids
from src.embeddings import embed_batch

CORPUS_PATH = PROJECT_ROOT / "corpus" / "ops_notes.json"


def ensure_index(conn, table: str) -> str:
    """Create the ANN index: StreamingDiskANN when vectorscale is available,
    otherwise a plain pgvector HNSW index. Returns which one was created."""
    index_name = f"{table}_embedding_idx"
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vectorscale'")
        has_vectorscale = cur.fetchone() is not None
        cur.execute(f"DROP INDEX IF EXISTS {index_name}")
        if has_vectorscale:
            cur.execute(
                f"CREATE INDEX {index_name} ON {table} "
                f"USING diskann (embedding vector_cosine_ops)"
            )
            return "StreamingDiskANN (pgvectorscale)"
        cur.execute(
            f"CREATE INDEX {index_name} ON {table} "
            f"USING hnsw (embedding vector_cosine_ops)"
        )
        return "HNSW (pgvector)"


def telemetry_window(conn, table: str) -> tuple[dt.datetime, dt.datetime] | None:
    with conn.cursor() as cur:
        cur.execute(f"SELECT min(ts), max(ts) FROM {table}")
        bounds = cur.fetchone()
    if bounds and bounds[0] and bounds[1]:
        return bounds[0], bounds[1]
    return None


def anchor_timestamps(
    notes: list[dict], window: tuple[dt.datetime, dt.datetime]
) -> None:
    """Spread note timestamps proportionally across the telemetry window so
    the hybrid time-series join has data to show in the demo."""
    start, end = window
    span = (end - start).total_seconds()
    count = len(notes)
    for position, note in enumerate(notes):
        offset = span * (position + 1) / (count + 1)
        note["logged_at"] = start + dt.timedelta(seconds=offset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchor-to-data",
        action="store_true",
        help="Remap note timestamps into the telemetry window.",
    )
    args = parser.parse_args()

    settings = load_settings()
    log_connect()

    # 1. Extensions + table. Extensions are attempted one at a time so a
    # console-managed extension degrades to the documented fallback instead
    # of aborting the bootstrap.
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
            if cur.fetchone() is None:
                raise RuntimeError(
                    "pgvector is not available on this service. Enable it from "
                    "the Tiger Data console (or use a service with pgvector)."
                )
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                print("Extension ready: vector")
            except psycopg.errors.InsufficientPrivilege:
                print(
                    "Extension 'vector' must be enabled from the Tiger Data "
                    "console (Service -> Extensions), then rerun init_db."
                )
                raise
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vectorscale")
                print("Extension ready: vectorscale (StreamingDiskANN index)")
            except psycopg.Error:
                conn.rollback()
                print(
                    "vectorscale unavailable — falling back to HNSW "
                    "(standard pgvector index)."
                )
        # The ops_notes table is created here (not in sql/init.sql) so the
        # embedding column dimension always matches EMBEDDING_DIM.
        with conn.cursor() as cur:
            cur.execute(
                f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = '{settings.ops_table}'
                          AND column_name = 'embedding'
                    ) THEN
                        CREATE TABLE {settings.ops_table} (
                            note_id        integer PRIMARY KEY,
                            substation_id  text        NOT NULL,
                            logged_at      timestamptz NOT NULL,
                            author         text        NOT NULL,
                            category       text        NOT NULL,
                            note           text        NOT NULL,
                            embedding      vector({settings.embedding_dim}) NOT NULL,
                            created_at     timestamptz NOT NULL DEFAULT now()
                        );
                    END IF;
                END $$;
                """
            )

    # 2. Load the corpus and resolve substation identifiers against the
    # telemetry table (auto-detects text vs numeric IDs and catalog tables).
    notes = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    corpus_ids = sorted({note["substation_id"] for note in notes})

    with connect() as conn:
        mapping, log_lines = resolve_substation_map(
            conn, settings.telemetry_table, corpus_ids, SUBSTATION_ID_MAP
        )
        for line in log_lines:
            print(line)
        for note in notes:
            note["substation_id"] = mapping.get(
                note["substation_id"], note["substation_id"]
            )
        if args.anchor_to_data:
            window = telemetry_window(conn, settings.telemetry_table)
            if window:
                anchor_timestamps(notes, window)
                print(
                    f"Anchored {len(notes)} notes into telemetry window "
                    f"{window[0]} -> {window[1]}"
                )
            else:
                print(
                    "Warning: telemetry table is empty; timestamps left unchanged."
                )
        present = set(substation_ids(conn, settings.telemetry_table))
        unknown = {n["substation_id"] for n in notes} - present
        if unknown and present:
            print(
                "Warning: these note substation IDs still have no telemetry rows: "
                + ", ".join(sorted(str(value) for value in unknown))
                + ". Search will show 'no telemetry rows in this window' for "
                + "those notes; fix SUBSTATION_ID_MAP in src/config.py."
            )

    # 3. Embed and upsert.
    texts = [note["note"] for note in notes]
    vectors, provider, model = embed_batch(texts, settings)
    print(f"Embedded {len(texts)} notes with provider={provider} model={model}")

    with connect() as conn:
        with conn.cursor() as cur:
            for note, vector in zip(notes, vectors, strict=True):
                cur.execute(
                    f"""
                    INSERT INTO {settings.ops_table}
                        (note_id, substation_id, logged_at, author, category, note, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (note_id) DO UPDATE SET
                        substation_id = EXCLUDED.substation_id,
                        logged_at     = EXCLUDED.logged_at,
                        author        = EXCLUDED.author,
                        category      = EXCLUDED.category,
                        note          = EXCLUDED.note,
                        embedding     = EXCLUDED.embedding
                    """,
                    (
                        note["note_id"],
                        note["substation_id"],
                        note["logged_at"],
                        note["author"],
                        note["category"],
                        note["note"],
                        vector,
                    ),
                )
            index_type = ensure_index(conn, settings.ops_table)
        cur2 = conn.cursor()
        cur2.execute(f"SELECT count(*) FROM {settings.ops_table}")
        total = cur2.fetchone()[0]
    print(f"Upserted {total} notes into {settings.ops_table}")
    print(f"ANN index: {index_type}")


if __name__ == "__main__":
    main()
