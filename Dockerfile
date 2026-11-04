# Modelo externo: la primera predicción requiere conexión para descargar pesos.
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/
WORKDIR /app
ENV UV_LINK_MODE=copy PYTHONUNBUFFERED=1 HF_HOME=/app/.cache/huggingface
# Incluye uv.lock cuando exista; uv respeta el lock al instalar.
COPY . .
RUN uv sync --no-dev --no-install-project
EXPOSE 8501
CMD ["uv", "run", "--no-dev", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
