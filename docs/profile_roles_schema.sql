alter table public.profiles
  add column if not exists role text not null default 'user';

alter table public.profiles
  drop constraint if exists profiles_role_check;

alter table public.profiles
  add constraint profiles_role_check check (role in ('user', 'admin'));

create index if not exists profiles_role_idx
  on public.profiles (role);

-- Promote a specific account to admin after signup.
-- Replace the email before running this.
--
-- update public.profiles
-- set role = 'admin'
-- where id = (
--   select id
--   from auth.users
--   where email = 'your-email@example.com'
-- );

-- Everyone else remains a normal learner profile:
--
-- role = 'user'
