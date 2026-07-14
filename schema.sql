DROP TABLE IF EXISTS documents;

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    embedding vector(1024) NOT NULL,

    source_file VARCHAR(255) NOT NULL,
    chunk_index INTEGER NOT NULL,
    total_chunks INTEGER,
    file_size_bytes INTEGER,

    created_at TIMESTAMP DEFAULT NOW(),

    metadata JSONB

);


CREATE INDEX ON documents USING ivfflat (emgedding vector_cosine_ops) WITH (lists = 10);

CREATE INDEX idx_source_file ON documents(source_file);