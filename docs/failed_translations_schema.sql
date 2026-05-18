create table if not exists public.failed_translations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  user_word_id uuid null references public.user_words(id) on delete set null,
  input_text text not null,
  model_normalized text null,
  model_translation text null,
  model_is_slang boolean null,
  model_metadata jsonb not null default '{}'::jsonb,
  user_feedback text not null default 'wrong',
  source text not null default 'app',
  expected_normalized text null,
  expected_translation text null,
  expected_is_slang boolean null,
  failure_type text null,
  status text not null default 'needs_review',
  created_at timestamptz not null default now(),
  reviewed_at timestamptz null,
  constraint failed_translations_status_check check (
    status in (
      'needs_review',
      'approved',
      'rejected',
      'added_to_eval',
      'added_to_training'
    )
  )
);

alter table public.failed_translations
  add column if not exists reviewed_at timestamptz null;

create index if not exists failed_translations_status_created_idx
  on public.failed_translations (status, created_at desc);

create index if not exists failed_translations_user_created_idx
  on public.failed_translations (user_id, created_at desc);

alter table public.failed_translations enable row level security;

create policy "Users can insert their own failed translations"
  on public.failed_translations
  for insert
  to authenticated
  with check (auth.uid() = user_id);

create policy "Users can read their own failed translations"
  on public.failed_translations
  for select
  to authenticated
  using (auth.uid() = user_id);
