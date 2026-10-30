# =============================================================================
# UAO-Neumonia — imagen de producción (Streamlit + TensorFlow CPU)
#
# Reglas (CONSTITUTION.md §2, §3, §5):
#   - uv es el ÚNICO gestor de dependencias; nunca `pip install -r requirements.txt`
#   - TensorFlow solo entra en Docker/CI (en local el proyecto corre sin TF, lazy)
#   - El checkpoint WilhemNet_86.h5 se MONTA en runtime, no se hornea en la imagen
# =============================================================================

FROM python:3.13-slim

# Bootstrap del binario uv (único uso de pip en la imagen: instalar el gestor).
RUN pip install --no-cache-dir uv

WORKDIR /app

# Capa de dependencias (cacheable): primero pyproject + lock, para no reinstalar
# TensorFlow ante cada cambio de código fuente.
COPY pyproject.toml uv.lock ./

# Runtime + grupo "docker" (tensorflow-cpu==2.20.0); SIN el grupo "dev".
RUN uv sync --no-dev --group docker

# Código de la aplicación y resto del proyecto (exclusiones en .dockerignore).
COPY . .

# El checkpoint se monta en runtime (-v $(pwd)/WilhemNet_86.h5:/app/WilhemNet_86.h5)
# y se referencia vía MODEL_PATH (sobrescribible con -e en el arranque).
ENV MODEL_PATH=/app/WilhemNet_86.h5

# Proyecto en "modo virtual" (sin paquete): la raíz /app debe estar en el path
# para que `from src.controllers...` funcione (Streamlit agrega solo el dir del script).
ENV PYTHONPATH=/app

EXPOSE 8501

CMD ["uv", "run", "--no-sync", "streamlit", "run", "src/views/app.py"]
