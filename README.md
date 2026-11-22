# grid-ops-search: semantic search over grid operations on Tiger Data

Grid-ops-search stores vector embeddings alongside time-series telemetry in one Tiger Data (TimescaleDB) database and joins them in SQL. The [VoltStream](https://github.com/princenzmw/voltstream) simulator streams substation load and voltage readings into a `grid_readings` hypertable. This project adds operational notes (maintenance logs, fault reports, switching operations) embedded with pgvector and indexed with pgvectorscale StreamingDiskANN in the same database, then answers natural-language queries by joining both tables:

> "voltage sag complaints" -> matching operational notes plus the 5-minute load and voltage telemetry around each event.

## Why store both in Postgres

Time-series rows record load and voltage drops; operational notes record the cause. Keeping the embedding column next to the time-series hypertable lets one SQL query jump from a semantic match to the busbar behavior around that timestamp.

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
| `corpus/ops_notes.json` | 14 substation operations notes across four substations |
| `src/config.py` | Environment-driven settings and masked URL logging |
| `src/db.py` | psycopg connection factory with pgvector type support |
| `src/embeddings.py` | Provider layer: OpenAI-compatible API or deterministic keyless hashing fallback |
| `src/init_db.py` | Schema bootstrap, embedding, idempotent upsert, ANN index creation |
| `src/inspect.py` | Prints telemetry table range and substation IDs for corpus alignment |
| `src/search.py` | Hybrid semantic + time-series search CLI |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # paste your Tiger Data connection string as DATABASE_URL
```

`DATABASE_URL` comes from the Tiger Data console (Services -> your service -> Connection info). Keep it out of screenshots; the repo masks it in all log output.

### 1. Substation identifier alignment (automatic)

The corpus uses readable slugs (`nairobi_west`, `industrial_area`, ...). Your telemetry table may store slugs in another format or use numeric IDs. `init_db` resolves this at load time with case-insensitive matching against stored `substation_id` values (ignoring underscores), including lookups through a substation catalog table (such as `substations(id, name)`) when the telemetry column is numeric. Resolution results print during the load.

If a corpus ID is reported as `UNRESOLVED` (the message prints the actual stored values), add an explicit override to `SUBSTATION_ID_MAP` in `src/config.py`. Run `python -m src.inspect` to inspect column types, stored IDs, catalog names, and telemetry range.

### 2. Load and embed the corpus

```bash
python -m src.init_db --anchor-to-data
```

- Embeds all 14 notes with the configured provider (selects OpenAI when `OPENAI_API_KEY` is set, otherwise the keyless hashing fallback) and upserts them into `ops_notes`.
- Creates a StreamingDiskANN index (`diskann` / `vector_cosine_ops`) when pgvectorscale is available, or falls back to an HNSW index.
- `--anchor-to-data` spreads note timestamps across the actual telemetry window so the hybrid join returns readings.

### 3. Search

```bash
python -m src.search "voltage sag complaints"
python -m src.search "transformer maintenance and oil"
python -m src.search "protection relay operation" --k 3 --window 45
```

Each match prints the note, its cosine similarity, and the 5-minute average load and voltage telemetry around the event.

## Embedding providers

| Provider | When | Behaviour |
| --- | --- | --- |
| OpenAI-compatible (`text-embedding-3-small`, 1536-dim) | `OPENAI_API_KEY` present, or `EMBEDDING_PROVIDER=openai` | Semantic similarity; works with any OpenAI-compatible endpoint via `OPENAI_API_BASE` |
| Feature hashing (`hashing`, 1536-dim) | No key, or `EMBEDDING_PROVIDER=hashing` | Deterministic, keyless word + character n-gram features, sublinear TF weighting, L2-normalized; exercises the same pgvector storage, indexing, and query path |

## SQL walk-through

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

MIT, see [LICENSE](LICENSE).
