-- Тихий момент: нет незавершённых remember-jobs и конвейеров за последние 2 часа.
with last as (
  select distinct on (pipeline_run_id) pipeline_run_id, pipeline_name, status, created_at
  from pipeline_runs
  where pipeline_run_id is not null and status is not null and created_at > now() - interval '2 hours'
  order by pipeline_run_id, created_at desc
)
select 'pipeline' as kind, pipeline_name, status::text, created_at from last
 where status::text like '%STARTED%' or status::text like '%INITIATED%'
union all
select 'remember_job', operation_name, 'running', started_at from pipeline_runs
 where operation_name = 'remember' and outcome is null and ended_at is null
   and started_at > now() - interval '2 hours'
order by 4;
