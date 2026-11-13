"""Print the telemetry facts needed to align the corpus with the database.

Shows the substation_id column type, the identifiers actually stored, their
catalog display names when a substation catalog table exists, and the
telemetry time range. Run this first whenever substation identifiers seem
misaligned.
"""

from __future__ import annotations

from src.config import load_settings
from src.db import (
    connect,
    substation_column_type,
    substation_ids,
)


def main() -> None:
    settings = load_settings()
    with connect() as conn:
        column_type = substation_column_type(conn, settings.telemetry_table)
        ids = substation_ids(conn, settings.telemetry_table)
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT min(ts), max(ts), count(*) FROM {settings.telemetry_table}"
            )
            first, last, total = cur.fetchone()

        print(f"Telemetry table : {settings.telemetry_table}")
        print(f"substation_id type: {column_type}")
        print(f"Rows            : {total:,}  |  range: {first} -> {last}")
        print("Stored substation_id values:")
        for value in ids:
            label = ""
            if column_type not in ("text", "character varying", "unknown"):
                label = _catalog_name(conn, value)
            suffix = f"  ({label})" if label else ""
            print(f"  - {value!r}{suffix}")
    print(
        "\ninit_db resolves corpus IDs automatically against these values; "
        "add explicit overrides to SUBSTATION_ID_MAP in src/config.py only "
        "if a value remains UNRESOLVED."
    )


def _catalog_name(conn, value) -> str:
    """Best-effort display name for a numeric substation id."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE column_name IN ('name', 'display_name', 'substation_name')
              AND table_schema = 'public'
            """
        )
        for table_name, name_column in cur.fetchall():
            try:
                cur.execute(
                    f'SELECT {name_column} FROM "{table_name}" WHERE id = %s',
                    (value,),
                )
                row = cur.fetchone()
            except Exception:
                continue
            if row and row[0]:
                return str(row[0])
    return ""


if __name__ == "__main__":
    main()
