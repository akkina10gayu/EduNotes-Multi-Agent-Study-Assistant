create or replace function match_document_chunks(
    query_embedding vector(384),
    match_threshold float,
    match_count int,
    p_user_id uuid
)
returns table (
    id uuid,
    document_id uuid,
    chunk_index int,
    content text,
    metadata jsonb,
    similarity float
)
language sql stable
as $$
    select
        dc.id,
        dc.document_id,
        dc.chunk_index,
        dc.content,
        dc.metadata,
        1 - (dc.embedding <=> query_embedding) as similarity
    from document_chunks dc
    join documents d on dc.document_id = d.id
    where d.user_id = p_user_id
        and 1 - (dc.embedding <=> query_embedding) > match_threshold
    order by dc.embedding <=> query_embedding
    limit match_count;
$$;
