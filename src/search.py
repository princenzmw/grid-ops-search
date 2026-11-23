"""Semantic search over operations notes, joined with live telemetry.

For each query the CLI:
  1. embeds the query with the configured provider,
  2. runs an approximate nearest-neighbour search over ops_notes using
     cosine distance (served by the StreamingDiskANN / HNSW index),
  3. pulls the 5-minute rollup of load and voltage from the VoltStream
     telemetry table around each matching note's timestamp,
  4. prints one panel per match: the note, its similarity score, and the
     surrounding telemetry window.

Usage:
    python -m src.search "voltage sag complaints"
    python -m src.search "transformer maintenance" --k 3 --window 45
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections.abc import Sequence

import psycopg
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import load_settings
from src.db import connect
from src.embeddings import embed_texts

console = Console()


def find_notes(conn, settings, query_vector, k: int) -> list[dict]:
    """ANN search over ops_notes with cosine distance."""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT note_id, substation_id, logged_at, author, category, note,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM {settings.ops_table}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_vector, query_vector, k),
        )
        columns = [desc.name for desc in cur.description]
        return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]


def telemetry_window(conn, settings, substation_id: str, centre: dt.datetime,
                     window_minutes: int) -> list[tuple]:
    """5-minute load/voltage rollup around a timestamp for one substation."""
    span = dt.timedelta(minutes=window_minutes)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT time_bucket(INTERVAL '5 minutes', ts) AS bucket,
                   round(avg(load_mw)::numeric, 2)      AS avg_load_mw,
                   round(avg(voltage_kv)::numeric, 2)   AS avg_voltage_kv
            FROM {settings.telemetry_table}
            WHERE substation_id = %s
              AND ts BETWEEN %s AND %s
            GROUP BY bucket
            ORDER BY bucket
            """,
            (substation_id, centre - span, centre + span),
        )
        return cur.fetchall()


def render(note: dict, rows: list[tuple]) -> None:
    meta = (
        f"[bold]note_id[/bold] {note['note_id']}   "
        f"[bold]substation[/bold] {note['substation_id']}   "
        f"[bold]logged_at[/bold] {note['logged_at']:%Y-%m-%d %H:%M %Z}   "
        f"[bold]author[/bold] {note['author']}   "
        f"[bold]category[/bold] {note['category']}"
    )
    similarity = float(note["similarity"])
    console.print(
        Panel(
            note["note"],
            title=f"similarity {similarity:.3f}",
            subtitle=meta,
            border_style="cyan" if similarity >= 0.5 else "white",
        )
    )
    if not rows:
        console.print(
            "  [yellow]No telemetry rows in this window "
            "(note timestamp outside the simulated data range).[/yellow]\n"
        )
        return
    table = Table(box=None, pad_edge=False)
    table.add_column("bucket (5 min)", style="bold")
    table.add_column("avg load (MW)", justify="right")
    table.add_column("avg voltage (kV)", justify="right")
    for bucket, load, voltage in rows:
        table.add_row(f"{bucket:%H:%M}", str(load), str(voltage))
    console.print(table, end="\n\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="natural-language search query")
    parser.add_argument("--k", type=int, default=4, help="number of matches")
    parser.add_argument(
        "--window", type=int, default=30, help="telemetry window in minutes"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit machine-readable JSON to stdout",
    )
    return parser


def format_match_payload(note: dict, rows: list[tuple]) -> dict:
    """Serialize one matched note and its telemetry rollup into JSON-safe types."""
    logged_at = note["logged_at"]
    logged_iso = (
        logged_at.isoformat() if isinstance(logged_at, dt.datetime) else str(logged_at)
    )
    telemetry = [
        {
            "bucket": bucket.isoformat()
            if isinstance(bucket, dt.datetime)
            else str(bucket),
            "avg_load_mw": float(load) if load is not None else None,
            "avg_voltage_kv": float(voltage) if voltage is not None else None,
        }
        for bucket, load, voltage in rows
    ]
    return {
        "note_id": note["note_id"],
        "substation_id": note["substation_id"],
        "logged_at": logged_iso,
        "author": note["author"],
        "category": note["category"],
        "note": note["note"],
        "similarity": round(float(note["similarity"]), 6),
        "telemetry": telemetry,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    settings = load_settings()
    result = embed_texts([args.query], settings)
    if not args.as_json:
        console.print(
            f"[bold]grid-ops-search[/bold]  provider={result.provider}  "
            f"model={result.model}  query={args.query!r}\n"
        )

    with connect() as conn:
        matches = find_notes(conn, settings, result.vector, args.k)
        if not matches:
            if args.as_json:
                print(
                    json.dumps(
                        {
                            "query": args.query,
                            "provider": result.provider,
                            "model": result.model,
                            "k": args.k,
                            "window_minutes": args.window,
                            "matches": [],
                        },
                        indent=2,
                    )
                )
            else:
                console.print(
                    "[red]No notes stored yet: run python -m src.init_db first.[/red]"
                )
            return

        payload_matches: list[dict] = []
        for note in matches:
            try:
                rows = telemetry_window(
                    conn, settings, note["substation_id"], note["logged_at"], args.window
                )
            except psycopg.errors.InvalidTextRepresentation:
                conn.rollback()
                if not args.as_json:
                    console.print(
                        "[yellow]The note's substation_id does not match the "
                        "telemetry table's column type. Notes were loaded with an "
                        "outdated mapping. Rerun:[/yellow] "
                        "[bold]python -m src.init_db --anchor-to-data[/bold] "
                        "[yellow](it auto-resolves corpus IDs against "
                        "grid_readings; see also python -m src.inspect)[/yellow]"
                    )
                rows = []
            if args.as_json:
                payload_matches.append(format_match_payload(note, rows))
            else:
                render(note, rows)

        if args.as_json:
            print(
                json.dumps(
                    {
                        "query": args.query,
                        "provider": result.provider,
                        "model": result.model,
                        "k": args.k,
                        "window_minutes": args.window,
                        "matches": payload_matches,
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
