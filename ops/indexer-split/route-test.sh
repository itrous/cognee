#!/bin/bash
# Проверка маршрутизации шлюза на ЗАГЛУШКАХ: nginx с боевым конфигом и два эхо-сервера
# с именами cognee и cognee-indexer в отдельной docker-сети. Живой стек и данные
# не участвуют. Код 0 — все пары «метод путь» ушли туда, куда ожидалось.
# По умолчанию проверяется ТОТ файл, который монтирует compose (CONF переопределяет),
# и он обязан совпадать с копией в репозитории.
set -u
cd "$(dirname "$0")"
NET=cognee-route-test-$$
MOUNTED=/home/itrous/cognee-patches/indexer-split/nginx.conf
CONF=${CONF:-$MOUNTED}
if [ "$CONF" = "$MOUNTED" ] && ! cmp -s "$MOUNTED" nginx.conf; then
  echo "FAIL смонтированный $MOUNTED отличается от ops/indexer-split/nginx.conf"; exit 1
fi
STUB='import http.server,os
class H(http.server.BaseHTTPRequestHandler):
    def _r(self):
        b=os.environ["NAME"].encode(); self.send_response(200); self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b)
    do_GET=do_POST=do_PUT=do_PATCH=do_DELETE=do_OPTIONS=_r
    def log_message(self,*a): pass
http.server.ThreadingHTTPServer(("0.0.0.0",8000),H).serve_forever()'
cleanup() { docker rm -f rt-api-$$ rt-idx-$$ rt-gw-$$ rt-cli-$$ >/dev/null 2>&1; docker network rm $NET >/dev/null 2>&1; }
trap cleanup EXIT
docker network create $NET >/dev/null
docker run -d --name rt-api-$$ --network $NET --network-alias cognee -e NAME=cognee_api --entrypoint python cognee/cognee:pio-server-aed21d907 -c "$STUB" >/dev/null
docker run -d --name rt-idx-$$ --network $NET --network-alias cognee-indexer -e NAME=cognee_indexer --entrypoint python cognee/cognee:pio-server-aed21d907 -c "$STUB" >/dev/null
docker run -d --name rt-gw-$$ --network $NET -v "$(realpath "$CONF"):/etc/nginx/nginx.conf:ro" nginx:1.29-alpine >/dev/null
# Клиент — curl из образа cognee (в busybox wget нет произвольных методов).
docker run -d --name rt-cli-$$ --network $NET --entrypoint sleep cognee/cognee:pio-server-aed21d907 infinity >/dev/null
# Готовность: шлюз отвечает и оба эхо-сервера достижимы через него. Ждём состояния, а не
# времени: проба повторяется, пока не получит оба ответа, без пауз и без счётчика попыток,
# поэтому исход не зависит от скорости машины. Событие, которое обрывает ожидание, — выход
# любого из контейнеров.
a=; b=
until [ "$a" = cognee_api ] && [ "$b" = cognee_indexer ]; do
  if [ "$(docker inspect -f '{{.State.Running}}' rt-api-$$ rt-idx-$$ rt-gw-$$ rt-cli-$$ | grep -c true)" != 4 ]; then
    echo "FAIL контейнер заглушки, шлюза или клиента завершился до готовности"
    docker logs rt-gw-$$ 2>&1 | tail -5
    exit 1
  fi
  a=$(docker exec rt-cli-$$ curl -sf -m 2 "http://rt-gw-$$:8000/health" 2>/dev/null)
  b=$(docker exec rt-cli-$$ curl -sf -m 2 -X POST "http://rt-gw-$$:8000/api/v1/cognify" 2>/dev/null)
done
fail=0; n=0
while read -r exp m p; do
  [ -z "$exp" ] && continue; n=$((n+1))
  got=$(docker exec rt-cli-$$ curl -sf -X "$m" "http://rt-gw-$$:8000$p" || echo ERR)
  [ "$got" = "$exp" ] || { fail=$((fail+1)); echo "FAIL $m $p -> $got (ждали $exp)"; }
done < routes.txt
echo "проверено $n, ошибок $fail"
[ $fail -eq 0 ]
