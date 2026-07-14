CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE knowledge_sources (
    knowledge_source text PRIMARY KEY,
    corpus_revision text
);

CREATE TABLE source_documents (
    knowledge_source text NOT NULL REFERENCES knowledge_sources (knowledge_source),
    document_id text NOT NULL,
    source_path text NOT NULL,
    source_revision text NOT NULL,
    processing_revision text NOT NULL,
    title text NOT NULL,
    raw_content text NOT NULL,
    PRIMARY KEY (knowledge_source, document_id),
    UNIQUE (knowledge_source, source_path)
);

CREATE TABLE indexed_chunks (
    chunk_id text PRIMARY KEY,
    knowledge_source text NOT NULL,
    document_id text NOT NULL,
    source_revision text NOT NULL,
    processing_revision text NOT NULL,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    heading_path text[] NOT NULL,
    content text NOT NULL,
    embedding vector NOT NULL,
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('english', content)
    ) STORED,
    UNIQUE (knowledge_source, document_id, ordinal),
    FOREIGN KEY (knowledge_source, document_id)
        REFERENCES source_documents (knowledge_source, document_id)
        ON DELETE CASCADE
);

CREATE INDEX indexed_chunks_search_vector_idx
    ON indexed_chunks USING gin (search_vector);

CREATE TABLE sync_reports (
    run_id uuid PRIMARY KEY,
    knowledge_source text NOT NULL,
    status text NOT NULL CHECK (status IN ('succeeded', 'failed')),
    started_at timestamptz NOT NULL,
    finished_at timestamptz NOT NULL,
    processing_revision text NOT NULL,
    observed_snapshot_size integer NOT NULL CHECK (observed_snapshot_size >= 0),
    added_count integer NOT NULL CHECK (added_count >= 0),
    updated_count integer NOT NULL CHECK (updated_count >= 0),
    deleted_count integer NOT NULL CHECK (deleted_count >= 0),
    unchanged_count integer NOT NULL CHECK (unchanged_count >= 0),
    ignored_count integer NOT NULL CHECK (ignored_count >= 0),
    failed_source_path text,
    error_summary text,
    corpus_revision text
);
