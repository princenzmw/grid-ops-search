# grid-ops-search — semantic search over grid operations, powered by pgvector + pgvectorscale on Tiger Data

Grid-ops-search stores **high-dimensional vector embeddings alongside time-series data** in a single Tiger Data (TimescaleDB) service, then joins them back together. It is the human-context layer for a live grid-telemetry workload: the [VoltStream](https://github.com/princenzmw/voltstream) simulator streams substation load and voltage readings into a `grid_readings` hypertable; this project adds operational notes — maintenance logs, fault reports, switching operations — embedded with **pgvector** and indexed with **pgvectorscale's StreamingDiskANN** in the *same database*, and answers natural-language queries by combining the two:

> "voltage sag complaints" → the matching ops notes **and** the 5-minute load/voltage telemetry around each event.

## Why this matters

Time-series tells you *what* happened; operational notes tell you *why*. Both live in Postgres here — the embeddings column sits next to the time-series columns, and a single SQL query walks from a semantic match to the exact telemetry window. This is the pattern utilities actually need: an operator types "voltage sag complaints" and gets the note *plus* the busbar behaviour around that moment.

## Architecture

```
corpus/ops_notes.json ──► src/init_db.py ──► embeddings (OpenAI-compatible or keyless hashing)
                                                │
                                                ▼
                        Tiger Data service (TimescaleDB / PostgreSQL)
                        ├── grid_readings      hypertable (VoltStream telemetry, 1-h chunks)
                        └── ops_notes          note text + vector(1536) embedding
                                               └── StreamingDiskANN index (pgvectorscale),
                                                   HNSW fallback when vectorscale is absent

python -m src.search ──► embed query ──► ANN search (cosine distance)
                                    └─► time_bucket() rollup of load/voltage
                                        around each matched note's timestamp
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `sql/init.sql` | Idempotent extension bootstrap (`vector`, `vectorscale`) |
| `corpus/ops_notes.json` | 14 realistic substation operations notes (four substations) |
| `src/config.py` | Environment-driven settings, credential hygiene (masked URLs) |
| `src/db.py` | psycopg connection factory with pgvector type support |
| `src/embeddings.py` | Provider layer: OpenAI-compatible API or deterministic keyless hashing fallback |
| `src/init_db.py` | Schema bootstrap, embedding, idempotent upsert, ANN index creation |
| `src/inspect.py` | Prints telemetry table range and substation IDs (for aligning the corpus) |
| `src/search.py` | Hybrid semantic + time-series search CLI |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # paste your Tiger Data connection string as DATABASE_URL
```

`DATABASE_URL` comes from the Tiger Data console (Services → your service → Connection info). Keep it out of screenshots — the repo never echoes it unmasked.

### 1. Substation identifier alignment (automatic)

The corpus uses human-readable slugs (`nairobi_west`, `industrial_area`, ...). Your telemetry table may store text slugs in another format or numeric IDs. `init_db` resolves this automatically at load time: case/underscore-insensitive matching against the stored `substation_id` values, including name lookups through a substation catalog table (e.g. `substations(id, name)`) when the telemetry column is numeric. Resolution results are printed during the load.

Only if a corpus ID is reported as `UNRESOLVED` (the message prints the actual stored values), add an explicit override to `SUBSTATION_ID_MAP` in `src/config.py`. `python -m src.inspect` shows the column type, stored IDs, catalog names and telemetry range at any time.

### 2. Load and embed the corpus

```bash
python -m src.init_db --anchor-to-data
```

- Embeds all 14 notes with the configured provider (auto-selects OpenAI when `OPENAI_API_KEY` is set, otherwise the keyless hashing fallback) and upserts them into `ops_notes`.
- Creates a **StreamingDiskANN** index (`diskann` / `vector_cosine_ops`) when pgvectorscale is available; otherwise a **HNSW** index.
- `--anchor-to-data` spreads the note timestamps across the actual telemetry window so the hybrid join has data to show.

### 3. Search

```bash
python -m src.search "voltage sag complaints"
python -m src.search "transformer maintenance and oil"
python -m src.search "protection relay operation" --k 3 --window 45
```

Each match prints the note, its cosine similarity, and the 5-minute average load/voltage telemetry around the event.

## Embedding providers

| Provider | When | Behaviour |
| --- | --- | --- |
| OpenAI-compatible (`text-embedding-3-small`, 1536-dim) | `OPENAI_API_KEY` present, or `EMBEDDING_PROVIDER=openai` | True semantic similarity; works with any OpenAI-compatible endpoint via `OPENAI_API_BASE` |
| Feature hashing (`hashing`, 1536-dim) | No key, or `EMBEDDING_PROVIDER=hashing` | Deterministic, keyless: word + character n-gram features, sublinear TF weighting, L2-normalised — lexical similarity that exercises the identical pgvector storage, indexing and query path |

## SQL walk-through (what the demo proves)

```sql
-- High-dimensional vectors stored next to time-series metadata
SELECT note_id, substation_id, logged_at, left(note, 60) AS note
FROM ops_notes
ORDER BY embedding <=> '[0.0103, -0.0421, ...]'::vector
LIMIT 4;                                    -- cosine ANN search

-- The hybrid join: telemetry around each matched event
SELECT time_bucket(INTERVAL '5 minutes', r.ts) AS bucket,
       avg(r.load_mw), avg(r.voltage_kv)
FROM grid_readings r
JOIN ops_notes n ON n.substation_id = r.substation_id
WHERE r.ts BETWEEN n.logged_at - INTERVAL '30 minutes'
               AND n.logged_at + INTERVAL '30 minutes'
GROUP BY bucket ORDER BY bucket;
```

## License

MIT — see [LICENSE](LICENSE).
