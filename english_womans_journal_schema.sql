CREATE TABLE chunks (
  id TEXT PRIMARY KEY,
  parent_id TEXT,
  chunk_index INTEGER,
  article_id TEXT,
  journal TEXT,
  journal_id TEXT,
  title TEXT,
  raw_title TEXT,
  author TEXT,
  section TEXT,
  volume TEXT,
  number TEXT,
  date TEXT,
  year INTEGER,
  page_no_start TEXT,
  page_no_end TEXT,
  page_label_start TEXT,
  page_label_end TEXT,
  text TEXT NOT NULL
, text_clean TEXT);
CREATE VIRTUAL TABLE chunks_fts
USING fts5(
  id UNINDEXED,
  title,
  text,
  tokenize='porter'
)
/* chunks_fts(id,title,text) */;
CREATE TABLE IF NOT EXISTS 'chunks_fts_data'(id INTEGER PRIMARY KEY, block BLOB);
CREATE TABLE IF NOT EXISTS 'chunks_fts_idx'(segid, term, pgno, PRIMARY KEY(segid, term)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS 'chunks_fts_content'(id INTEGER PRIMARY KEY, c0, c1, c2);
CREATE TABLE IF NOT EXISTS 'chunks_fts_docsize'(id INTEGER PRIMARY KEY, sz BLOB);
CREATE TABLE IF NOT EXISTS 'chunks_fts_config'(k PRIMARY KEY, v) WITHOUT ROWID;
CREATE TABLE embeddings (
  id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
  dim INTEGER NOT NULL,
  vec BLOB NOT NULL
);
CREATE INDEX idx_chunks_year ON chunks(year);
CREATE INDEX idx_chunks_journal ON chunks(journal_id);
