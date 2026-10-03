-- HybridRAG initial schema: one PostgreSQL store for hybrid retrieval.
-- The migration runner substitutes {{EMBEDDING_DIM}} with the dimension of
-- the configured embedding model before applying this file.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE documents (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_path  text NOT NULL UNIQUE,
    title        text NOT NULL DEFAULT '',
    content      text NOT NULL,
    content_hash text NOT NULL,
    ingested_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id bigint NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    ordinal     integer NOT NULL,
    content     text NOT NULL,
    embedding   vector({{EMBEDDING_DIM}}) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (document_id, ordinal)
);

-- Dense retrieval: approximate nearest neighbor index (cosine distance).
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);

-- Lexical retrieval: inverted index over the generated tsvector.
CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv);

-- Fuzzy lexical fallback: trigram index on the raw chunk text.
CREATE INDEX chunks_content_trgm ON chunks USING gin (content gin_trgm_ops);
