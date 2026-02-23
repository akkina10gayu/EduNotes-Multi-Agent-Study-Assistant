-- Enable pgvector extension
create extension if not exists vector;

-- ==================== KNOWLEDGE BASE ====================

create table documents (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users(id) on delete cascade,
    title text not null,
    topic text,
    source text,
    url text,
    content text not null,
    metadata jsonb default '{}',
    created_at timestamptz default now(),
    updated_at timestamptz default now()
);

create table document_chunks (
    id uuid primary key default gen_random_uuid(),
    document_id uuid references documents(id) on delete cascade,
    chunk_index integer not null,
    content text not null,
    embedding vector(384),
    metadata jsonb default '{}',
    created_at timestamptz default now()
);

-- ==================== FLASHCARDS ====================

create table flashcard_sets (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users(id) on delete cascade,
    topic text not null,
    source_content text,
    card_count integer default 0,
    created_at timestamptz default now()
);

create table flashcards (
    id uuid primary key default gen_random_uuid(),
    set_id uuid references flashcard_sets(id) on delete cascade,
    front text not null,
    back text not null,
    difficulty text default 'medium',
    times_reviewed integer default 0,
    times_correct integer default 0,
    last_reviewed timestamptz,
    created_at timestamptz default now()
);

-- ==================== QUIZZES ====================

create table quizzes (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users(id) on delete cascade,
    topic text not null,
    source_content text,
    question_count integer default 0,
    created_at timestamptz default now()
);

create table quiz_questions (
    id uuid primary key default gen_random_uuid(),
    quiz_id uuid references quizzes(id) on delete cascade,
    question text not null,
    question_type text default 'multiple_choice',
    options jsonb,
    correct_answer text not null,
    explanation text,
    order_index integer
);

create table quiz_attempts (
    id uuid primary key default gen_random_uuid(),
    quiz_id uuid references quizzes(id) on delete cascade,
    user_id uuid references auth.users(id) on delete cascade,
    started_at timestamptz default now(),
    completed_at timestamptz,
    score numeric,
    total_questions integer,
    correct_count integer
);

create table quiz_answers (
    id uuid primary key default gen_random_uuid(),
    attempt_id uuid references quiz_attempts(id) on delete cascade,
    question_id uuid references quiz_questions(id),
    user_answer text,
    is_correct boolean,
    answered_at timestamptz default now()
);

-- ==================== PROGRESS ====================

create table study_activities (
    id uuid primary key default gen_random_uuid(),
    user_id uuid references auth.users(id) on delete cascade,
    activity_type text not null,
    topic text,
    metadata jsonb default '{}',
    created_at timestamptz default now()
);

create table study_streaks (
    user_id uuid primary key references auth.users(id) on delete cascade,
    current_streak integer default 0,
    best_streak integer default 0,
    last_activity_date date,
    updated_at timestamptz default now()
);

-- ==================== INDEXES ====================

create index idx_documents_user on documents(user_id);
create index idx_documents_topic on documents(topic);
create index idx_document_chunks_doc on document_chunks(document_id);
create index idx_flashcard_sets_user on flashcard_sets(user_id);
create index idx_quizzes_user on quizzes(user_id);
create index idx_activities_user on study_activities(user_id);
create index idx_activities_date on study_activities(created_at);

create index idx_chunks_embedding on document_chunks
    using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- ==================== ROW LEVEL SECURITY ====================

alter table documents enable row level security;
alter table document_chunks enable row level security;
alter table flashcard_sets enable row level security;
alter table flashcards enable row level security;
alter table quizzes enable row level security;
alter table quiz_questions enable row level security;
alter table quiz_attempts enable row level security;
alter table quiz_answers enable row level security;
alter table study_activities enable row level security;
alter table study_streaks enable row level security;

create policy "Users access own data" on documents
    for all using (auth.uid() = user_id);
create policy "Users access own data" on flashcard_sets
    for all using (auth.uid() = user_id);
create policy "Users access own data" on quizzes
    for all using (auth.uid() = user_id);
create policy "Users access own data" on quiz_attempts
    for all using (auth.uid() = user_id);
create policy "Users access own data" on study_activities
    for all using (auth.uid() = user_id);
create policy "Users access own data" on study_streaks
    for all using (auth.uid() = user_id);

create policy "Users access own chunks" on document_chunks
    for all using (
        document_id in (select id from documents where user_id = auth.uid())
    );
create policy "Users access own flashcards" on flashcards
    for all using (
        set_id in (select id from flashcard_sets where user_id = auth.uid())
    );
create policy "Users access own questions" on quiz_questions
    for all using (
        quiz_id in (select id from quizzes where user_id = auth.uid())
    );
create policy "Users access own answers" on quiz_answers
    for all using (
        attempt_id in (select id from quiz_attempts where user_id = auth.uid())
    );
