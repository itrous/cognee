-- Тихий момент перед остановкой процесса, исполняющего конвейеры.
-- Запуск: psql -v since="'<StartedAt контейнера>'" -f quiet.sql
-- Строка, начатая ДО старта нынешнего процесса, не может в нём исполняться: она
-- помечена stale и тишине не мешает (такие закрывает стартовое восстановление,
-- remember-job не закрывает никто). Всё начатое после :since — активно, без
-- ограничения по возрасту.
-- Ограничение: строку, начатую после :since другим процессом, который затем умер
-- (так 2026-10-08 остался cognify d72eb1c1 старого API, начатый во время его
-- остановки), запрос считает ACTIVE. Сверка: исполняющий процесс пишет
-- «Pipeline run started: `<id>`» в свой лог — нет id в логе живого процесса,
-- значит строка мертва; её закроет стартовое восстановление через час.
-- Слепое пятно: операции, у которых нет строки до завершения (`record_operation`
-- пишет её на выходе) — forget, DELETE-маршруты, импорт skills (`add_skills` без
-- pipeline run), начало remember content_type=code. Их защищает только graceful
-- остановка: gunicorn ждёт запросы в полёте 30 с, stop_grace_period 40 с.
with last as (
  select distinct on (pipeline_run_id) pipeline_run_id, pipeline_name, status, created_at
  from pipeline_runs
  where pipeline_run_id is not null and status is not null
  order by pipeline_run_id, created_at desc
), open_rows as (
  select 'pipeline' as kind, pipeline_name as name, created_at as started from last
   where status::text like '%STARTED%' or status::text like '%INITIATED%'
  union all
  select 'remember_job', operation_name, started_at from pipeline_runs
   where operation_name = 'remember' and outcome is null and ended_at is null
)
select kind, name, started, 'ACTIVE' as state from open_rows
 where started >= :since::timestamptz
union all
select 'stale (до :since)', kind, max(started), count(*)::text from open_rows
 where started < :since::timestamptz group by kind
order by 3;
-- Тихо, если нет ни одной строки ACTIVE.
