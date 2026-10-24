# ==============================================================================
# Dockerfile Oficial - UAO-Neumonia
# Entorno reproducible basado en Python 3.13 + UV + OpenCV
# ==============================================================================

# 1. Imagen base oficial ligera de Python 3.13
FROM python:3.13-slim

# 2. Copiar el binario oficial de UV (sin instalar nada de PIP)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Instalar dependencias del sistema operativo (OpenCV, Tkinter, Make)
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    libgl1 \
    libglib2.0-0 \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# 4. Directorio de trabajo dentro del contenedor
WORKDIR /app

# 5. Copiar primero la definición de dependencias para aprovechar el caché de Docker
COPY pyproject.toml uv.lock ./

# 6. Sincronizar el entorno virtual con UV de forma determinista
RUN uv sync --frozen --no-install-project --no-cache

# 7. Copiar el código fuente y demás archivos del proyecto
COPY . .

# 8. Variable de entorno para que Python reconozca la carpeta src/ y habilite Keras 2
ENV PYTHONPATH=/app
ENV TF_USE_LEGACY_KERAS=1

# 9. Comando por defecto: ejecutar la suite de pruebas unitarias
CMD ["uv", "run", "pytest", "test/", "-v"]
