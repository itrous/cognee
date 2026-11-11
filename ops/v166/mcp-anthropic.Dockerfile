# Накладка: anthropic==1.8.0 в образ cognee-mcp (LLM_PROVIDER=anthropic).
FROM cognee/cognee-mcp:pio-v166-fa157b89
RUN /app/.venv/bin/uv pip install --python /app/.venv/bin/python --no-cache "anthropic==1.8.0"
