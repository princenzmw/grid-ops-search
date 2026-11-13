"""Database access helpers for grid-ops-search."""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Iterator

import psycopg
from pgvector.psycopg import register_vector

from src.config import load_settings, masked_url


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    """Open a connection with pgvector type support registered."""
    settings = load_settings()
    conn = psycopg.connect(settings.database_url, autocommit=False)
    try:
        register_vector(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def log_connect() -> None:
    settings = load_settings()
    print(f"Connecting to Tiger Data service ({masked_url(settings.database_url)})")


def substation_ids(conn: psycopg.Connection, telemetry_table: str) -> list[str]:
    """Distinct substation identifiers present in the telemetry table."""
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT DISTINCT substation_id FROM {telemetry_table} ORDER BY 1"
        )
        return [row[0] for row in cur.fetchall()]


def substation_column_type(conn: psycopg.Connection, telemetry_table: str) -> str:
    """Data type of the substation_id column (e.g. 'text' or 'integer')."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT data_type FROM information_schema.columns
            WHERE table_name = %s AND column_name = 'substation_id'
            """,
            (telemetry_table,),
        )
        row = cur.fetchone()
    return row[0] if row else "unknown"


def _normalise(label: object) -> str:
    """'Nairobi West' / 'nairobi_west' / 'nairobi-west 33kV' -> 'nairobi west'."""
    return re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip()


def resolve_substation_map(
    conn: psycopg.Connection,
    telemetry_table: str,
    corpus_ids: list[str],
    static_map: dict[str, str],
) -> tuple[dict[str, str], list[str]]:
    """Resolve corpus substation identifiers to the identifiers the telemetry
    table actually stores.

    Returns (mapping, log_lines). The mapping covers every corpus id that
    could be resolved; ids listed in the explicit SUBSTATION_ID_MAP always
    win. Resolution strategy:

    1. explicit SUBSTATION_ID_MAP entries,
    2. exact (case/underscore-insensitive) match against distinct
       substation_id values,
    3. for numeric columns: name lookup in a catalog table such as
       `substations` (id + name columns), matched the same insensitive way,
    4. substring match as a last resort.
    """
    mapping = dict(static_map)
    lines: list[str] = []
    column_type = substation_column_type(conn, telemetry_table)
    present = substation_ids(conn, telemetry_table)

    # Candidate display names for numeric ids: from any catalog table that
    # has an id column and a name-like column.
    catalog: dict[str, str] = {}  # normalised name -> str(id)
    if column_type not in ("text", "character varying", "unknown"):
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name, column_name
                FROM information_schema.columns
                WHERE column_name IN ('name', 'display_name', 'substation_name')
                  AND table_schema = 'public'
                """
            )
            candidates = cur.fetchall()
        for table_name, name_column in candidates:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        f'SELECT id, {name_column} FROM "{table_name}"'
                    )
                    rows = cur.fetchall()
            except psycopg.Error:
                continue
            for row_id, name in rows:
                normalised = _normalise(name)
                if normalised:
                    catalog.setdefault(normalised, str(row_id))
            if catalog:
                break

    lookup_exact = {_normalise(value): str(value) for value in present}
    unresolved: list[str] = []
    for corpus_id in corpus_ids:
        if corpus_id in mapping:
            continue
        wanted = _normalise(corpus_id)
        if wanted in lookup_exact:
            mapping[corpus_id] = lookup_exact[wanted]
            lines.append(f"resolved {corpus_id!r} -> {mapping[corpus_id]!r} (exact)")
            continue
        if wanted in catalog:
            mapping[corpus_id] = catalog[wanted]
            lines.append(
                f"resolved {corpus_id!r} -> id {mapping[corpus_id]} (catalog)"
            )
            continue
        partial = None
        for candidate, value in list(lookup_exact.items()) + list(catalog.items()):
            if wanted in candidate or candidate in wanted:
                partial = value
                break
        if partial is not None:
            mapping[corpus_id] = partial
            lines.append(
                f"resolved {corpus_id!r} -> {partial!r} (partial match)"
            )
        else:
            unresolved.append(corpus_id)

    if unresolved:
        lines.append(
            "UNRESOLVED: " + ", ".join(unresolved)
            + f" — the telemetry table stores these substation_id values: "
            + ", ".join(str(value) for value in present)
            + f" (column type {column_type}). Add them to SUBSTATION_ID_MAP "
            + "in src/config.py, e.g. " + ", ".join(
                f"{key!r}: {value!r}" for key, value in list(zip(unresolved, present))[:4]
            )
        )

    # Numeric columns need native ints, not numeric strings, in INSERT/SELECT
    # parameters.
    if column_type not in ("text", "character varying", "unknown"):
        for key, value in list(mapping.items()):
            if isinstance(value, str) and value.lstrip("-").isdigit():
                mapping[key] = int(value)
    return mapping, lines
