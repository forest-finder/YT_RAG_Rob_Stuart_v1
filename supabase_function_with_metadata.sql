-- Create a new stored function that returns content chunks with video metadata
CREATE OR REPLACE FUNCTION match_content_chunks_with_metadata(
  query_embedding vector(1536),
  match_threshold float,
  match_count int
)
RETURNS TABLE (
  id uuid,
  chunk_id varchar,
  video_id varchar,
  context text,
  content text,
  start_time varchar,
  end_time varchar,
  created_at timestamptz,
  updated_at timestamptz,
  similarity float,
  video_url varchar,
  title text,
  channel_name varchar,
  channel_handle varchar,
  view_count varchar,
  likes varchar,
  keywords text,
  is_shorts bool,
  comments text,
  date_published timestamptz
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    cc.id,
    cc.chunk_id,
    cc.video_id,
    cc.context,
    cc.content,
    cc.start_time,
    cc.end_time,
    cc.created_at,
    cc.updated_at,
    1 - (cc.embedding <=> query_embedding) AS similarity,
    vm.video_url,
    vm.title,
    vm.channel_name,
    vm.channel_handle,
    vm.view_count,
    vm.likes,
    vm.keywords,
    vm.is_shorts,
    vm.comments,
    vm.date_published
  FROM content_chunks cc
  LEFT JOIN video_metadata vm ON cc.video_id = vm.video_id
  WHERE 1 - (cc.embedding <=> query_embedding) > match_threshold
  ORDER BY cc.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;