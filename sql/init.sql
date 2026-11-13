-- grid-ops-search — schema bootstrap
-- Run automatically by `python -m src.init_db`; also safe to paste into the
-- Tiger Data SQL Editor. Idempotent: every statement uses IF NOT EXISTS.

-- Core pgvector support.
CREATE EXTENSION IF NOT EXISTS vector;

-- pgvectorscale adds the StreamingDiskANN index access method. Tiger Cloud
-- ships it; if your service does not have it yet, the init_db script falls
-- back to a plain pgvector HNSW index automatically.
CREATE EXTENSION IF NOT EXISTS vectorscale;

-- The ops_notes table is created by src/init_db.py so the embedding column
-- dimension always matches EMBEDDING_DIM from your .env (default 1536).
