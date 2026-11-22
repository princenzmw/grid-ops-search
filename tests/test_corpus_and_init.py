"""Tests for the operations note corpus, substation normalization, and timestamp anchoring."""

from __future__ import annotations

import datetime as dt
import json

from src.db import _normalise
from src.init_db import CORPUS_PATH, anchor_timestamps


def test_corpus_has_14_valid_notes() -> None:
    notes = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    assert len(notes) == 14
    ids = [n["note_id"] for n in notes]
    assert len(set(ids)) == 14

    required = {"note_id", "substation_id", "logged_at", "author", "category", "note"}
    for item in notes:
        assert required.issubset(item.keys())
        assert str(item["note"]).strip()
        assert str(item["substation_id"]).strip()
        dt.datetime.fromisoformat(item["logged_at"])


def test_normalise_matches_slugs_and_display_names() -> None:
    assert _normalise("Nairobi West") == "nairobi west"
    assert _normalise("nairobi_west") == "nairobi west"
    assert _normalise("Industrial-Area") == "industrial area"


def test_anchor_timestamps_places_notes_strictly_within_window() -> None:
    notes = [{"note_id": i} for i in range(1, 6)]
    start = dt.datetime(2026, 11, 12, 8, 0, tzinfo=dt.UTC)
    end = dt.datetime(2026, 11, 12, 14, 0, tzinfo=dt.UTC)

    anchor_timestamps(notes, (start, end))
    stamps = [n["logged_at"] for n in notes]
    assert all(start < ts < end for ts in stamps)
    assert stamps == sorted(stamps)
    assert len(set(stamps)) == len(stamps)
