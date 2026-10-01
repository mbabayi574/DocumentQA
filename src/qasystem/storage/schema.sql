-- plan.md §6. SQLite is the single source of truth; Chroma holds no document text.
--
-- Every retrieval path reads chunks through `eligible_chunks` and nothing else, so a
-- stale or deleted row is unreachable even if Chroma cleanup never ran (I1, I2).

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
  doc_id         TEXT PRIMARY KEY,
  title          TEXT NOT NULL,
  source_name    TEXT NOT NULL,
  format         TEXT NOT NULL CHECK (format IN ('pdf', 'txt', 'md')),
  status         TEXT NOT NULL CHECK (status IN ('active', 'deleted')),
  current_version INTEGER,                       -- NULL until first publish, and while deleted
  last_version   INTEGER NOT NULL DEFAULT 0,     -- monotonic allocator; never decreases
  content_hash   TEXT,
  parsed_hash    TEXT,                           -- sha256(raw bytes) / sha256(parsed text)
  language_hint  TEXT,
  created_at     TEXT NOT NULL,
  updated_at     TEXT NOT NULL,
  published_at   TEXT,
  deleted_at     TEXT
);

CREATE TABLE IF NOT EXISTS document_versions (
  doc_id        TEXT NOT NULL REFERENCES documents(doc_id),
  version       INTEGER NOT NULL,
  content_hash  TEXT NOT NULL,
  source_text   TEXT NOT NULL,                   -- parsed text, so I6 is verifiable offline
  state         TEXT NOT NULL CHECK (state IN ('staging', 'published', 'superseded', 'failed')),
  created_at    TEXT NOT NULL,
  published_at  TEXT,
  PRIMARY KEY (doc_id, version)
);

-- chunk_id is also the Chroma vector id: "{doc_id}:v{version}:{ordinal}" (§4.1), so a
-- stale vector can never collide with a fresh one.
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id          TEXT PRIMARY KEY,
  doc_id            TEXT NOT NULL,
  doc_version       INTEGER NOT NULL,
  ordinal           INTEGER NOT NULL,
  text              TEXT NOT NULL,
  char_start        INTEGER NOT NULL,
  char_end          INTEGER NOT NULL,
  chunk_hash        TEXT NOT NULL,
  embed_input_hash  TEXT NOT NULL,               -- sha256 of the exact embedded string
  section_path_json TEXT NOT NULL,
  page_start        INTEGER,
  page_end          INTEGER,
  line_start        INTEGER,
  line_end          INTEGER,
  language          TEXT NOT NULL CHECK (language IN ('fa', 'en', 'mixed')),
  UNIQUE (doc_id, doc_version, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_by_version ON chunks (doc_id, doc_version);

-- Lexical index. `tokens` holds OUR tokenizer's output, ZWNJ split components
-- included, so SQLite's own tokenizer never has to understand Persian. Standalone
-- (not external-content) so rows for staging versions exist too; eligibility is
-- applied by the join, never by what is present here.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5 (
  chunk_id UNINDEXED,
  tokens
);

-- Vectors keyed by the exact embedded string plus the model that produced them.
-- The model_id in the key is what stops two models sharing vectors (I9); the
-- unchanged-string key is what makes re-ingest free (I5) and rebuild API-free (I10).
CREATE TABLE IF NOT EXISTS embedding_cache (
  input_hash TEXT NOT NULL,
  model_id   TEXT NOT NULL,
  dim        INTEGER NOT NULL,
  vector     BLOB NOT NULL,                      -- float32, little-endian
  created_at TEXT NOT NULL,
  PRIMARY KEY (input_hash, model_id)
);

-- Counts and timings only: no bodies, no text, no secrets.
CREATE TABLE IF NOT EXISTS ingest_log (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id         TEXT NOT NULL,
  action         TEXT NOT NULL,
  prev_version   INTEGER,
  new_version    INTEGER,
  chunks_added   INTEGER NOT NULL DEFAULT 0,
  chunks_reused  INTEGER NOT NULL DEFAULT 0,
  chunks_removed INTEGER NOT NULL DEFAULT 0,
  embed_requests INTEGER NOT NULL DEFAULT 0,
  duration_ms    INTEGER,
  status         TEXT NOT NULL,
  error_code     TEXT,
  ts             TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ingest_log_by_doc ON ingest_log (doc_id, id);

-- THE choke point. Every retrieval path reads chunks ONLY through this view.
-- source_name comes along so retrieval reads the text and everything a citation quotes in
-- ONE statement: no second query, so no window between deciding eligibility and using it.
CREATE VIEW IF NOT EXISTS eligible_chunks AS
  SELECT c.*, d.source_name AS source_name FROM chunks c
  JOIN documents d ON d.doc_id = c.doc_id
                  AND d.status = 'active'
                  AND d.current_version = c.doc_version;
