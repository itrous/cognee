# План: вынести индексацию из процесса API cognee (стек pio)

## Предмет
Стек `compose.stack.yml` (вне git). Сейчас `cognee-api` — один процесс gunicorn
`-w 1` (захардкожено в `entrypoint.sh`), в одном event loop крутятся и фоновые
конвейеры (add+cognify+improve), и recall/session/trace. Под индексацией
session/trace отвечают 0,2–1,3 с вместо ~15–25 мс, /health до 1,8 с, graph-recall
изредка вылетает за таймаут.

## Факты из кода (основание выбора)
1. Блокировки датасета — in-process: `cognee/infrastructure/locks/dataset_lock.py`
   (docstring: «process-local only (asyncio) — this does NOT protect against multiple
   processes/workers»). Используют: `modules/pipelines/operations/pipeline.py:165`
   (все конвейеры), `api/v1/remember/remember.py:1659`, `api/v1/forget/forget.py:298,399`,
   `api/v1/datasets/datasets.py:198,244` (удаления), `api/v1/update/incremental.py:604`,
   `modules/integrations/slack/history.py:153`.
2. Блокировка improve и сессий — in-process: `cognee/infrastructure/locks/session_lock.py`
   («Scope: single-worker FastAPI»). improve: `api/v1/improve/improve.py:303`.
3. Апстрим (`.agents/skills/cognee-performance/SKILL.md` §8): OSS масштабируется
   вертикально в одном процессе, `gunicorn -w 1`, распределённого раннера нет.
4. Состояние, которое ДОЛЖНО быть общим, — в общих СУБД: статус фонового remember —
   строка `pipeline_runs` («readable from any API worker», `api/v1/remember/remember_job.py`);
   сессионный кэш — `cache.db` (SQLite WAL) на общем томе, кросс-процессная
   корректность — транзакции `SqlCacheAdapter` (busy timeout 30 с, FOR UPDATE);
   этот же том и те же СУБД уже делит с API второй процесс — `cognee-mcp` в прямом режиме.
5. `/api/v1/remember/entry` с записями `qa`/`trace`/`feedback` пишет только в сессионный
   кэш и improve не запускает (поле `self_improvement` в `get_remember_router.py`). Исключение —
   запись `skill_run`: `remember_skill_run_entry` пишет узел в граф (`add_data_points`,
   `modules/tools/skill_runs.py`), а с `skill_improvement` ещё и `improve_skill_from_config`
   (`api/v1/remember/remember.py:441-480`). Ни то, ни другое `dataset_lock` не берёт —
   ни сейчас, ни после разделения, поэтому координация с конвейерами от размещения не
   меняется. Развести `/remember/entry` по типу записи nginx не может (тип — в теле), а
   клиенты стека (`pi-cognee`, демон pio) `skill_run` не шлют.
6. JWT-секреты (`FASTAPI_USERS_*_SECRET`) заданы в `.env` — одинаковы во всех процессах.

## Решение
Вариант (а) «несколько воркеров gunicorn» отвергнут фактами 1–3. Выбран (б):
**маршрутизация по пути** — все операции, которые берут `dataset_lock` / improve-lock
или запускают конвейер, уходят в ОДИН отдельный процесс-индексатор; всё прочее —
в прежний API. Каждая in-process блокировка (`dataset_lock`, improve-lock) остаётся
внутри одного процесса, как и сейчас. Граф после разделения пишут: индексатор (все
конвейеры, improve, удаления), `cognee-mcp` (как и сейчас) и API — только через
`/remember/entry` с записью `skill_run`: узел `SkillRun` и, с `skill_improvement`, узел
`SkillImprovementProposal` (`add_data_points([proposal])`, `modules/memify/skill_improvement.py:159`)
и правка узла скилла при `apply`. Всё это и сейчас идёт мимо `dataset_lock` (см. факт 5).

Фоновые задачи, которые запускаются не HTTP-запросом, а при старте приложения
(`cognee/api/client.py` lifespan), маршрутизацией не управляются:
- Slack history scheduler (`modules/integrations/slack/history_sync.py:175`) стартует в
  каждом экземпляре при `SLACK_CLIENT_ID` и `SLACK_HISTORY_SYNC_ENABLED` (по умолчанию true) и
  берёт `dataset_lock`. Сейчас `SLACK_*` не заданы и `integration_credentials` пуста —
  планировщик не запускается нигде. Чтобы он и в будущем жил только в индексаторе, API
  получает `SLACK_HISTORY_SYNC_ENABLED=false` в `environment` compose.
- `recover_stale_pipeline_runs_on_startup` (`modules/cognify/recovery.py`) при старте
  любого процесса закрывает как брошенные только конвейеры старше
  `COGNEE_STALE_RUN_RECOVERY_MIN_AGE_SECONDS` (3600 с) — механизм рассчитан на соседние
  процессы. Конвейер индексатора, идущий дольше часа, рестарт API откатил бы; у нас
  конвейер длится ~1 мин. Тот же риск уже есть с `cognee-mcp`.

Компоненты (тот же образ `cognee/cognee:pio-server-aed21d907`, тот же `.env`, те же тома
и монтирования патчей halfvec):
- `cognee` (`cognee-api`) — без изменений, кроме публикации портов (хост-порт 8000
  отдаётся шлюзу; для диагностики `127.0.0.1:8010`) и `SLACK_HISTORY_SYNC_ENABLED=false`.
- `cognee-indexer` — новый, команда та же (entrypoint с миграциями под
  `pg_advisory_lock`), хост-порт только `127.0.0.1:8012`.
- `cognee-gateway` — `nginx:1.29-alpine`, публикует `127.0.0.1:8000` и `172.25.0.50:8000`,
  конфиг `ops/indexer-split/nginx.conf`.

Маршруты в индексатор (метод + путь, остальное — в API):
- POST `/api/v1/remember` (ровно этот путь, не `/remember/entry`), `/add`, `/cognify`,
  `/memify`, `/improve`, `/forget`, `/skills`, `/sync`, `/responses/`
- PATCH `/api/v1/update`
- DELETE `/api/v1/delete`, `/api/v1/datasets`, `/api/v1/datasets/{id}`,
  `/api/v1/datasets/{id}/data/{data_id}`, `/api/v1/skills/{skill_id}` (удаляет узел графа
  и эмбеддинги, `api/v1/skills/list_skills.py:93-135`)
- GET `/api/v1/remember/jobs/{id}` (читается из БД, но так статус и запуск в одном месте)
- POST `/api/v1/slack/history/{team}/import`, PUT `/api/v1/slack/history/{team}/sync`,
  POST `/api/v1/integrations/{provider}/sync`, POST `/api/v1/integrations/{provider}/events`
  (импорт истории под `dataset_lock`), DELETE `/api/v1/integrations/{provider}/connection`
  (с `delete_data` вызывает `forget`), GET `/api/v1/integrations/{provider}/callback`
  (`on_installed` — начальная синхронизация), POST `/api/v1/slack/(commands|events|interactive)`
  (remember и импорт истории)
- GET `/api/v1/cognify/subscribe/{run_id}` — WebSocket прогресса конвейера: очередь
  `pipeline_run_info_queues` живёт в памяти процесса, исполняющего конвейер. Шлюз
  пробрасывает `Upgrade`/`Connection` (иначе любой WebSocket через него — 404;
  `/visualize/subscribe` опрашивает БД и остаётся в API).

nginx: `client_max_body_size 0`, `proxy_read_timeout/send_timeout 3600s`,
`proxy_buffering off`, HTTP/1.1 keepalive к апстримам, `Host` и `X-Forwarded-*`
пробрасываются. Healthcheck шлюза — `/health` через него в API.

UI (`NEXT_PUBLIC_LOCAL_API_URL=http://cognee:8000`) серверными маршрутами ходит в API
напрямую — это только чтение (visualize, schema). Браузер UI ходит на хост :8000 = шлюз.

## Инварианты (проверяемые)
I1. Маршрутизация проверяется на ЗАГЛУШКАХ, без живого стека и без запросов к данным:
    `ops/indexer-split/route-test.sh` поднимает в отдельной docker-сети nginx с боевым
    `nginx.conf` и два эхо-сервера с именами `cognee` и `cognee-indexer`, затем гоняет
    `ops/indexer-split/routes.txt` (58 пар «метод путь → ожидаемый процесс»): все маршруты
    таблицы, включая параметризованные DELETE и строку запроса, и методы API на тех же URI
    (`GET/POST /api/v1/datasets`, `GET /api/v1/skills/{id}`, `PUT /datasets/{id}/schema`,
    `OPTIONS /remember`, DELETE чужих префиксов). Решение не зависит от режима
    аутентификации: приложение в проверке не участвует. Отрицательный контроль: мутант
    конфига (снят `DELETE /skills/*`, `GET|POST /datasets` уведены в индексатор) обязан дать
    ошибки — проверено: 4 ошибки, код 1. На живом стеке запросы к маршрутам удаления,
    `/forget`, `/update`, `/remember`, `/add`, `/cognify`, `/improve` без ключа или с ключом
    ради проверки маршрутизации не отправляются; живую сторону закрывают I2–I4 по реальному
    трафику.
I2. В access-логе `cognee-api` после переключения нет ни одной строки, совпадающей с
    `"(POST /api/v1/(remember|add|cognify|memify|improve|forget|skills|sync|responses/?)[ ?]|PATCH /api/v1/update[ ?]|DELETE /api/v1/(delete|datasets|skills)[/ ?]|GET /api/v1/remember/jobs/|(POST|PUT) /api/v1/(slack/history/[^/ ]+/(import|sync)|integrations/[^/ ]+/(sync|events))[ ?]|DELETE /api/v1/integrations/[^/ ]+/connection[ ?])`
    (у POST — якорь `[ ?]`, чтобы `/remember/entry` не совпадал; у DELETE — `[/ ?]`, чтобы
    ловились и `/datasets/{id}`, `/datasets/{id}/data/{data_id}`, `/skills/{skill_id}`).
    Положительный контроль regex — строки-образцы каждого маршрута таблицы (включая
    параметризованные DELETE) обязаны совпасть, а `POST /api/v1/remember/entry`,
    `GET /api/v1/skills/` — нет; плюс прогон по логу API до переключения находит совпадения.
I3. В логе `cognee-api` нет «Pipeline run started» (с v1.6.5 — «Pipeline item started») после переключения (конвейеры идут
    только в индексаторе); в логе `cognee-indexer` они есть.
I4. 26+ реальных конвейеров через :8000 (датасет `perf_probe_20261008`, только добавление) —
    все job `completed`, без `improve_error`.
I5. 4 параллельные сессии remember/entry→recall при 1 и 2 одновременных конвейерах:
    все области ответили, свой QA найден, каждый ход ≤ 8 с.
I6. Под индексацией (1 и 2 конвейера): session/trace p50 и p90 — десятки мс, max < 200 мс
    (до: p50 18–177 мс, p90 до 552 мс, max до 2,1 с); `/health` p90 < 50 мс, max < 200 мс
    (до: p90 до 216 мс, max до 1,8 с). Graph-recall p50/p90 — сравнить с «до»
    (1,5–3,5 с / 1,7–4,9 с), ухудшения нет; таймаутов graph-recall нет.
I7. Откат одной командой: `cp compose.stack.yml.bak-before-indexer-split-20261008
    compose.stack.yml && docker compose -f compose.stack.yml up -d --remove-orphans`.

## Порядок
1. Замер «до» (сделан): idle, 1 конвейер.
2. Резервная копия compose, правки, `docker compose config` — валидно.
3. Переключение в две фазы, чтобы окно обрыва было минимальным и видимым:
   а) `up -d cognee-indexer` — API не трогается, индексатор стартует без хост-порта 8000
      (миграции под `pg_advisory_lock`), ждём healthy. Трафик ещё целиком в API.
   б) тихий момент по БД: `ops/indexer-split/quiet.sql` (только чтение, `-v since=` — время
      старта останавливаемого процесса) не даёт ни одной строки ACTIVE — нет конвейера с
      последним статусом `DATASET_PROCESSING_STARTED/INITIATED` и нет remember-job
      (`operation_name='remember'`, `outcome IS NULL`), начатых после старта процесса, без
      ограничения по возрасту; начатое раньше старта в нём исполняться не может; это
      покрывает и окно «job записана, задача не создана», и фазу improve после cognify.
   в) сразу после — `docker compose stop -t 60 cognee`: SIGTERM, gunicorn перестаёт
      принимать соединения и даёт текущим запросам завершиться (graceful_timeout gunicorn
      30 с), lifespan дренирует фоновые задачи (8 с); затем `up -d` пересоздаёт API без
      хост-порта и поднимает шлюз. Недоступность :8000 — секунды до healthy API.
   Остаточный риск, принятый явно: синхронная операция без строки в БД до завершения
   (`forget`, DELETE — `record_operation` пишет строку на выходе), начатая в секундном окне
   между (б) и (в) и длящаяся дольше 30 с, будет оборвана. За последние 9 ч в access-логе
   API нет ни одного `forget`/DELETE от клиентов (только 401 от моих проб без ключа): pio
   и плагины их не шлют. Это тот же риск, что у любого рестарта, разрешённого задачей.
   I8. После переключения: тот же запрос по jobs, начатым до переключения, пуст; в логе
   остановки старого API нет «Shutting down with background tasks still running» и нет
   сообщений gunicorn о принудительном завершении воркера (`SIGKILL`, `WORKER TIMEOUT`):
   значит, запросы в полёте завершились в пределах graceful_timeout. Access-лог для этого
   не годится — строка пишется по завершении запроса.
4. Проверки I1–I6, замер «после».

## Известные остаточные риски
- in-memory OTEL-спаны (`/api/v1/activity/spans`) у API больше не содержат спанов
  конвейеров (трейсинг на стенде выключен).
- Сам graph-recall — CPU-bound в event loop API (1,5–4 с CPU на запрос: распаковка
  соседства neo4j, разбор halfvec); параллельные graph-recall по-прежнему делят одно ядро.
  Это отдельное узкое место, этим планом не лечится.
