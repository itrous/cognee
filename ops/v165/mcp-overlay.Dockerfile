# Накладка образа cognee-mcp: wheel cognee из ветки P и anthropic для LLM_PROVIDER=anthropic.
# Контекст сборки — каталог с одним cognee-1.6.5-py3-none-any.whl; база передаётся аргументом:
#   DOCKER_BUILDKIT=0 docker build --build-arg BASE_IMAGE=cognee/cognee-mcp:base-1.6.5-<P12> \
#     -f ops/v165/mcp-overlay.Dockerfile -t cognee/cognee-mcp:pio-v165-<P12> <каталог wheel>
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY cognee-1.6.5-py3-none-any.whl /tmp/
RUN /app/.venv/bin/uv pip install --python /app/.venv/bin/python --no-cache --no-deps --reinstall /tmp/cognee-1.6.5-py3-none-any.whl
RUN /app/.venv/bin/uv pip install --python /app/.venv/bin/python --no-cache "anthropic==1.8.0"
