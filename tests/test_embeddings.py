"""Tests for embedding normalization, determinism, and retrieval ordering."""

from __future__ import annotations

import json
import math

import pytest

from src.config import Settings
from src.embeddings import embed_batch, embed_texts
from src.init_db import CORPUS_PATH


def _hashing_settings(dim: int = 1536) -> Settings:
    return Settings(
        database_url="postgres://u:p@localhost/tsdb",
        embedding_provider="hashing",
        openai_api_key="",
        embedding_model="text-embedding-3-small",
        embedding_dim=dim,
        telemetry_table="grid_readings",
        ops_table="ops_notes",
    )


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hashing_provider_is_deterministic_and_unit_norm() -> None:
    settings = _hashing_settings()
    res1 = embed_texts(["voltage sag at industrial area"], settings)
    res2 = embed_texts(["voltage sag at industrial area"], settings)

    assert res1.provider == "hashing"
    assert res1.model == "feature-hashing-1536"
    assert len(res1.vector) == 1536
    assert res1.vector == res2.vector
    norm = math.sqrt(sum(v * v for v in res1.vector))
    assert math.isclose(norm, 1.0, rel_tol=1e-6)


def test_hashing_handles_empty_or_punctuation_text() -> None:
    settings = _hashing_settings(dim=128)
    vectors, provider, _ = embed_batch(["--- !!! ???"], settings)
    assert provider == "hashing"
    assert len(vectors[0]) == 128
    assert all(v == 0.0 for v in vectors[0])


def test_embed_batch_and_texts_empty_input() -> None:
    settings = _hashing_settings()
    assert embed_batch([], settings) == ([], "none", "none")
    with pytest.raises(ValueError, match="empty batch"):
        embed_texts([], settings)


def test_hashing_ranks_voltage_sag_note_above_unrelated_note() -> None:
    settings = _hashing_settings()
    notes = {n["note_id"]: n["note"] for n in json.loads(CORPUS_PATH.read_text(encoding="utf-8"))}

    q = embed_texts(["voltage sag complaints"], settings).vector
    note_voltage = embed_texts([notes[1]], settings).vector
    note_vegetation = embed_texts([notes[4]], settings).vector

    assert _cosine(q, note_voltage) > _cosine(q, note_vegetation)
