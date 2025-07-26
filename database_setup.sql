-- Terminal RAG System Database Setup
-- Run this in your Supabase SQL Editor

-- 1. Enable vector extension (if not already done)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create the vector search function for content_chunks table
CREATE OR REPLACE FUNCTION match_content_chunks(
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.5,
    match_count int DEFAULT 10
)
RETURNS TABLE(
    id uuid,
    chunk_id varchar,
    video_id varchar,
    content text,
    context text,
    start_time varchar,
    end_time varchar,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        content_chunks.id,
        content_chunks.chunk_id,
        content_chunks.video_id,
        content_chunks.content,
        content_chunks.context,
        content_chunks.start_time,
        content_chunks.end_time,
        1 - (content_chunks.embedding <=> query_embedding) AS similarity
    FROM content_chunks
    WHERE 1 - (content_chunks.embedding <=> query_embedding) > match_threshold
    ORDER BY content_chunks.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- 3. Test your setup
-- SELECT COUNT(*) FROM content_chunks WHERE embedding IS NOT NULL;