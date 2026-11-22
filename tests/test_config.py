"""Tests for environment settings and credential masking."""

from __future__ import annotations

import pytest

from src.config import Settings, load_settings, masked_url


def test_masked_url_redacts_password() -> None:
    raw = "postgres://tsdbadmin:s3cr3tPass@host.example.tsdb.cloud.tigerdata.com:5432/tsdb?sslmode=require"
    masked = masked_url(raw)
    assert "s3cr3tPass" not in masked
    assert masked == "postgres://tsdbadmin:****@host.example.tsdb.cloud.tigerdata.com:5432/tsdb?sslmode=require"


def test_masked_url_handles_urls_without_password() -> None:
    assert masked_url("postgres://user@localhost/db") == "postgres://user@localhost/db"
    assert masked_url("sqlite:///local.db") == "sqlite:///local.db"


def test_effective_provider_resolution() -> None:
    base = {
        "database_url": "postgres://u:p@h/db",
        "embedding_model": "text-embedding-3-small",
        "embedding_dim": 1536,
        "telemetry_table": "grid_readings",
        "ops_table": "ops_notes",
    }
    assert Settings(embedding_provider="auto", openai_api_key="", **base).effective_provider == "hashing"
    assert Settings(embedding_provider="auto", openai_api_key="sk-test", **base).effective_provider == "openai"
    assert Settings(embedding_provider="hashing", openai_api_key="sk-test", **base).effective_provider == "hashing"

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        _ = Settings(embedding_provider="openai", openai_api_key="", **base).effective_provider


def test_load_settings_requires_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "")
    with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
        load_settings()


def test_load_settings_reads_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pw@localhost/tsdb")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "HASHING")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("EMBEDDING_DIM", "384")
    monkeypatch.setenv("TELEMETRY_TABLE", "readings_v2")
    monkeypatch.setenv("OPS_TABLE", "notes_v2")

    s = load_settings()
    assert s.database_url == "postgres://user:pw@localhost/tsdb"
    assert s.embedding_provider == "hashing"
    assert s.effective_provider == "hashing"
    assert s.embedding_dim == 384
    assert s.telemetry_table == "readings_v2"
    assert s.ops_table == "notes_v2"
