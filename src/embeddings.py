"""Embedding providers for grid-ops-search.

Two providers, one interface:

- ``openai``  — real semantic embeddings (OpenAI-compatible API; works with
  OpenAI, Azure OpenAI gateways, and any OpenAI-compatible endpoint via
  OPENAI_API_BASE). Recommended: text-embedding-3-small, 1536 dimensions.
- ``hashing`` — keyless fallback: deterministic feature-hashing vectors over
  word and character n-grams with sublinear term weighting and L2
  normalisation. Suitable for demonstrating the pgvector storage, indexing and
  similarity-search mechanics without any external API; similarity is lexical
  rather than semantic.

Both return plain Python floats of length EMBEDDING_DIM, normalised to unit
length so cosine similarity and inner product are equivalent.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

import requests

from src.config import Settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NGRAM_SIZES = (2, 3, 4)  # character n-gram sizes for the hashing provider


@dataclass(frozen=True)
class EmbeddingResult:
    vector: list[float]
    provider: str
    model: str


# ---------------------------------------------------------------------------
# OpenAI-compatible provider
# ---------------------------------------------------------------------------

def _embed_openai(texts: list[str], settings: Settings) -> list[list[float]]:
    url = f"{settings.openai_api_base}/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    vectors: list[list[float]] = []
    batch_size = 64
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = requests.post(
            url,
            headers=headers,
            json={"model": settings.embedding_model, "input": batch},
            timeout=60,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Embedding API returned HTTP {response.status_code}: {response.text[:300]}"
            )
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        vectors.extend(item["embedding"] for item in data)
    return [_l2_normalise(vec) for vec in vectors]


# ---------------------------------------------------------------------------
# Hashing provider (keyless fallback)
# ---------------------------------------------------------------------------

def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _feature_weight(count: float) -> float:
    """Sublinear term frequency scaling, as in BM25 / TF-IDF."""
    return 1.0 + math.log(count) if count > 0 else 0.0


def _bucket(feature: str, dim: int) -> int:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dim


def _sign(feature: str) -> float:
    digest = hashlib.blake2b(b"sign:" + feature.encode("utf-8"), digest_size=8).digest()
    return 1.0 if digest[0] % 2 == 0 else -1.0


def _embed_hashing_one(text: str, dim: int) -> list[float]:
    vec = [0.0] * dim
    words = _tokens(text)
    features: list[str] = []

    # Word-level features carry most of the lexical signal.
    features.extend(f"w:{word}" for word in words)
    # Character n-grams add robustness to morphological variants
    # ("sag" vs "sags", "outage" vs "outages").
    for word in words:
        padded = f"^{word}$"
        for size in _NGRAM_SIZES:
            features.extend(f"c{size}:{padded[i : i + size]}" for i in range(len(padded) - size + 1))

    counts: dict[str, float] = {}
    for feature in features:
        counts[feature] = counts.get(feature, 0.0) + 1.0
    for feature, count in counts.items():
        idx = _bucket(feature, dim)
        vec[idx] += _sign(feature) * _feature_weight(count)
    return _l2_normalise(vec)


def _embed_hashing(texts: list[str], settings: Settings) -> list[list[float]]:
    return [_embed_hashing_one(text, settings.embedding_dim) for text in texts]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def _l2_normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def embed_texts(texts: list[str], settings: Settings) -> EmbeddingResult:
    """Embed a batch of texts with the configured provider."""
    if not texts:
        raise ValueError("embed_texts called with an empty batch")
    vectors, provider, model = embed_batch(texts, settings)
    return EmbeddingResult(vector=vectors[0], provider=provider, model=model)


def embed_batch(texts: list[str], settings: Settings) -> tuple[list[list[float]], str, str]:
    """Embed a batch; returns (vectors, provider, model)."""
    if not texts:
        return [], "none", "none"
    provider = settings.effective_provider
    if provider == "openai":
        return _embed_openai(texts, settings), "openai", settings.embedding_model
    return _embed_hashing(texts, settings), "hashing", "feature-hashing-1536"
