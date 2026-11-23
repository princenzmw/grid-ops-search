"""Tests for CLI argument parsing and JSON output in src.search."""

from __future__ import annotations

import datetime as dt
import json
from contextlib import nullcontext
from decimal import Decimal

import pytest

from src import search
from src.config import Settings
from src.embeddings import EmbeddingResult


def _dummy_settings() -> Settings:
    return Settings(
        database_url="postgres://u:p@localhost:5432/db",
        embedding_provider="hashing",
        openai_api_key="",
        embedding_model="text-embedding-3-small",
        embedding_dim=1536,
        telemetry_table="grid_readings",
        ops_table="ops_notes",
    )


def test_build_parser_defaults_and_flags() -> None:
    parser = search.build_parser()
    defaults = parser.parse_args(["voltage sag"])
    assert defaults.query == "voltage sag"
    assert defaults.k == 4
    assert defaults.window == 30
    assert defaults.as_json is False

    custom = parser.parse_args(["relay trip", "--k", "2", "--window", "45", "--json"])
    assert custom.query == "relay trip"
    assert custom.k == 2
    assert custom.window == 45
    assert custom.as_json is True


def test_format_match_payload_serializes_datetime_and_decimal() -> None:
    logged_at = dt.datetime(2026, 6, 15, 14, 30, tzinfo=dt.UTC)
    bucket = dt.datetime(2026, 6, 15, 14, 25, tzinfo=dt.UTC)
    note = {
        "note_id": 7,
        "substation_id": "nairobi_west",
        "logged_at": logged_at,
        "author": "ops_lead",
        "category": "voltage_quality",
        "note": "Evening peak flicker reported.",
        "similarity": Decimal("0.3571234"),
    }
    rows = [(bucket, Decimal("41.25"), Decimal("10.98"))]

    payload = search.format_match_payload(note, rows)
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)

    assert decoded["note_id"] == 7
    assert decoded["logged_at"] == logged_at.isoformat()
    assert decoded["similarity"] == 0.357123
    assert decoded["telemetry"] == [
        {
            "bucket": bucket.isoformat(),
            "avg_load_mw": 41.25,
            "avg_voltage_kv": 10.98,
        }
    ]

    empty_payload = search.format_match_payload(note, [])
    assert empty_payload["telemetry"] == []


def test_main_json_outputs_matches(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    logged_at = dt.datetime(2026, 6, 15, 8, 15, tzinfo=dt.UTC)
    bucket = dt.datetime(2026, 6, 15, 8, 10, tzinfo=dt.UTC)
    fake_note = {
        "note_id": 1,
        "substation_id": "industrial_area",
        "logged_at": logged_at,
        "author": "shift_eng",
        "category": "voltage_quality",
        "note": "Three customers reported morning voltage sags.",
        "similarity": 0.4125,
    }

    monkeypatch.setattr(search, "load_settings", _dummy_settings)
    monkeypatch.setattr(
        search,
        "embed_texts",
        lambda texts, settings: EmbeddingResult(
            vector=[0.1, 0.2], provider="hashing", model="feature-hashing-1536"
        ),
    )
    monkeypatch.setattr(search, "connect", lambda: nullcontext(object()))
    monkeypatch.setattr(search, "find_notes", lambda conn, settings, vec, k: [fake_note])
    monkeypatch.setattr(
        search,
        "telemetry_window",
        lambda conn, settings, sub, centre, window: [(bucket, Decimal("55.40"), Decimal("10.85"))],
    )

    search.main(["voltage sag", "--k", "2", "--window", "45", "--json"])
    out = capsys.readouterr().out
    parsed = json.loads(out)

    assert parsed["query"] == "voltage sag"
    assert parsed["provider"] == "hashing"
    assert parsed["model"] == "feature-hashing-1536"
    assert parsed["k"] == 2
    assert parsed["window_minutes"] == 45
    assert len(parsed["matches"]) == 1
    assert parsed["matches"][0]["substation_id"] == "industrial_area"
    assert parsed["matches"][0]["telemetry"][0]["avg_load_mw"] == 55.4


def test_main_json_handles_no_matches(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(search, "load_settings", _dummy_settings)
    monkeypatch.setattr(
        search,
        "embed_texts",
        lambda texts, settings: EmbeddingResult(
            vector=[0.1, 0.2], provider="hashing", model="feature-hashing-1536"
        ),
    )
    monkeypatch.setattr(search, "connect", lambda: nullcontext(object()))
    monkeypatch.setattr(search, "find_notes", lambda conn, settings, vec, k: [])

    search.main(["empty query", "--json"])
    parsed = json.loads(capsys.readouterr().out)

    assert parsed["query"] == "empty query"
    assert parsed["matches"] == []
