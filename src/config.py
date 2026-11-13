"""Central configuration for grid-ops-search.

All runtime configuration comes from environment variables (see .env.example).
No credentials are hard-coded anywhere in this repository.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env if present; real deployments inject env vars directly.
load_dotenv(PROJECT_ROOT / ".env")

# Explicit overrides for corpus -> telemetry substation identifiers. Leave
# empty: init_db auto-resolves corpus IDs against the telemetry table
# (case/underscore-insensitive name matching, including catalog tables).
# Add entries only if auto-resolution reports UNRESOLVED ids, using the
# actual values printed by `python -m src.inspect`, e.g.:
#   {"nairobi_west": 1, "industrial_area": 2}
SUBSTATION_ID_MAP: dict[str, str] = {}


@dataclass(frozen=True)
class Settings:
    database_url: str
    embedding_provider: str  # "auto" | "openai" | "hashing"
    openai_api_key: str
    embedding_model: str
    embedding_dim: int
    telemetry_table: str
    ops_table: str
    openai_api_base: str = field(default="https://api.openai.com/v1")

    @property
    def effective_provider(self) -> str:
        """Resolve 'auto' to the best available provider."""
        if self.embedding_provider == "openai":
            if not self.openai_api_key:
                raise RuntimeError(
                    "EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is not set."
                )
            return "openai"
        if self.embedding_provider == "hashing":
            return "hashing"
        # auto: use OpenAI when a key is present, else the keyless fallback
        return "openai" if self.openai_api_key else "hashing"


def load_settings() -> Settings:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env and paste the "
            "connection string for your Tiger Data service."
        )
    return Settings(
        database_url=database_url,
        embedding_provider=os.environ.get("EMBEDDING_PROVIDER", "auto").strip().lower(),
        openai_api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small").strip(),
        embedding_dim=int(os.environ.get("EMBEDDING_DIM", "1536")),
        telemetry_table=os.environ.get("TELEMETRY_TABLE", "grid_readings").strip(),
        ops_table=os.environ.get("OPS_TABLE", "ops_notes").strip(),
        openai_api_base=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1").strip(),
    )


def masked_url(url: str) -> str:
    """Return the connection string with the password masked (safe for logs)."""
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    if "@" not in rest:
        return url
    userinfo, host = rest.rsplit("@", 1)
    if ":" in userinfo:
        user = userinfo.split(":", 1)[0]
        return f"{scheme}://{user}:****@{host}"
    return url
