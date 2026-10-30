# UAO-Neumonia

[![CI](https://github.com/JuanC0319/UAO-Neumonia/actions/workflows/ci.yml/badge.svg)](https://github.com/JuanC0319/UAO-Neumonia/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.13-3776AB)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fastral-sh%2Fuv%2Fmain%2Fassets%2Fbadge.json)](https://docs.astral.sh/uv/)
[![pytest](https://img.shields.io/badge/tested%20with-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Ruff](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fastral-sh%2Fruff%2Fmain%2Fassets%2Fbadge%2Fv2.json)](https://docs.astral.sh/ruff/)
[![Licencia](https://img.shields.io/badge/licencia-MIT-green)](LICENSE)
[![Commits](https://img.shields.io/badge/commits-Conventional%20-fe5100)](https://www.conventionalcommits.org/)

> **Hola y bienvenido a la herramienta para la detección rápida de neumonía.**
> Deep Learning aplicado al procesamiento de imágenes radiográficas de tórax en
> formato DICOM, para clasificarlas en **3 categorías**: Neumonía Bacteriana,
> Neumonía Viral y Sin Neumonía — con **Grad-CAM** para resaltar con un mapa de
> calor las regiones relevantes de la imagen de entrada.

---

## 📑 Tabla de contenidos

- [Estado del proyecto](#-estado-del-proyecto)
- [Stack](#-stack)
- [Primeros pasos](#-primeros-pasos)
- [Uso](#-uso)
- [Docker](#-docker)
- [Arquitectura (MVC)](#-arquitectura-mvc)
- [Calidad](#-calidad)
- [Acerca del modelo](#-acerca-del-modelo)
- [Acerca de Grad-CAM](#-acerca-de-grad-cam)
- [Contribución](#-contribución)
- [Licencia y créditos](#-licencia-y-créditos)

---

## 🚧 Estado del proyecto

Refactor MVC en curso en la rama `feature/refactor-mvc` (PRs hacia `develop`,
revisados por el code reviewer @luisfermezaramrez-stack):

- ✅ **Capa modelo + controlador** verificados: recepción (`verify_pipeline`),
  pytest, Ruff y `py_compile` en verde.
- ✅ **Estandarización**: `uv` único, `Makefile`, pre-commit, Ruff, CI, licencia
  MIT y [arquitectura versionada con Archify](docs/arquitectura.html).
- ✅ **Vista Streamlit** (`src/views/app.py`) completada: sube imagen, predice
  (`Integrator.analyze`) y muestra el overlay Grad-CAM (`Integrator.explain`),
  con errores amigables y sin importar TensorFlow/OpenCV/pydicom.
- ✅ **Docker** (`Dockerfile` `python:3.13-slim` + `uv`): TensorFlow solo dentro de la
  imagen y checkpoint montado en runtime — ver sección [Docker](#-docker).
- 🚧 **Pendientes**: merge a `develop`, reporte PDF.

La **fuente única de verdad** del proyecto es [`CONSTITUTION.md`](CONSTITUTION.md);
el contrato de trabajo del equipo vive en [`CONVENCIONES.md`](CONVENCIONES.md).

## 🧰 Stack

| Categoría | Tecnología |
|-----------|------------|
| Lenguaje | Python **3.13** (`.python-version`) |
| Gestor de paquetes y entornos | [`uv`](https://docs.astral.sh/uv/) — **único** (`pip`/`conda` prohibidos) |
| Pruebas | `pytest` con dobles + fixtures (patrón AAA; corren **sin TensorFlow**) |
| Lint y formato | [Ruff](https://docs.astral.sh/ruff/) (line-length 100) |
| Higiene de commits | `pre-commit` (Ruff + higiene de archivos + `uv-lock`) |
| Tareas | `Makefile` (`make check` = lint + formato + recepción + pruebas) |
| CI | [GitHub Actions](.github/workflows/ci.yml) — paridad con `make check` |
| Visión / DICOM | OpenCV `4.12.0.88` headless · pydicom · Pillow |
| Deep Learning | TensorFlow — **solo Docker/CI**; carga diferida (*lazy*) en local |

## 🚀 Primeros pasos

> **Regla dura:** toda gestión de dependencias y entornos pasa por
> [`uv`](https://docs.astral.sh/uv/). `pip`, `conda` y `requirements.txt` están
> prohibidos.

1. Instale `uv`:

   ```bash
   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Clone el repositorio y entre a la carpeta:

   ```bash
   git clone https://github.com/JuanC0319/UAO-Neumonia.git
   cd UAO-Neumonia
   ```

3. Cree el entorno determinista desde `pyproject.toml` + `uv.lock` e instale los
   hooks:

   ```bash
   make dev          # equivalente a: uv sync + uv run pre-commit install
   ```

4. Copie el modelo entrenado `WilhemNet_86.h5` a la raíz del proyecto (pesa
   ~112 MB y **no se versiona**; lo comparte el equipo por unidad externa).

5. Verifique que el entorno está sano:

   ```bash
   make check        # lint + formato + recepción (7 checks) + pruebas
   ```

> Nota: en macOS Intel TensorFlow no tiene rueda oficial y queda fuera del entorno
> local. El modelo se carga de forma diferida (patrón **TF-lazy**) y TensorFlow real
> solo entra en juego dentro de Docker/CI.

## 🖥️ Uso

La interfaz gráfica está en `src/views/app.py` (Streamlit) y habla únicamente con
el `Integrator` — nunca importa `tensorflow`/`cv2`/`pydicom`:

```bash
uv run streamlit run src/views/app.py
```

El flujo de la UI: ingresar el nombre completo y el número de identificación del
paciente → subir una radiografía (`.dcm`, `.jpg`, `.jpeg` o `.png`) → *'Analizar'*
(deshabilitado mientras falten los datos) → se muestran la preview, el overlay
Grad-CAM y el botón *'Descargar informe PDF'* (encabezado
«Detector de Neumonía», datos del paciente, fecha y hora, estudio, puntaje más alto
del modelo e impresión diagnóstica de apoyo por IA). Los errores (archivo inexistente o corrupto, formato no soportado)
se informan de forma amigable. La persistencia CSV del monolito queda **fuera del
MVP** (no-propósito en `CONSTITUTION.md` §1).

El pipeline también se recibe por consola:

```bash
make verify        # 7 checks: lectura DICOM/JPG, preprocessing, integrador, errores presupuesto…
```

## 🐳 Docker

TensorFlow (CPU) vive **solo dentro de la imagen** (`tensorflow-cpu==2.20.0`, grupo
`docker` de `pyproject.toml`); en local el proyecto corre sin TF (patrón **TF-lazy**).
El checkpoint `WilhemNet_86.h5` se **monta en runtime**, nunca se hornea en la imagen.

1. Construir la imagen (`python:3.13-slim` + `uv`; `uv sync --no-dev --group docker`):

   ```bash
   docker build -t uao-neumonia .
   ```

2. Ejecutar la app con el checkpoint montado y el puerto expuesto:

   ```bash
   docker run --rm -p 8501:8501 \
     -v "$(pwd)/WilhemNet_86.h5:/app/WilhemNet_86.h5" \
     -e MODEL_PATH=/app/WilhemNet_86.h5 \
     uao-neumonia
   ```

   Luego abrí `http://localhost:8501`.

- `MODEL_PATH` está fijado por defecto a `/app/WilhemNet_86.h5` en la imagen; si montás
  el checkpoint en otra ruta, pasá `-e MODEL_PATH=/ruta/del/checkpoint.h5`.
- El `.dockerignore` excluye `.git`, `.venv`, el `.h5`, `data/`, `odd/`, cachés y
  documentación regenerable, para que el contexto de build sea liviano.

3. Eliminar la instancia cuando ya no se necesite:

   ```bash
   # parar y borrar el contenedor (si se arrancó con --name uao-neumonia-app)
   docker rm -f uao-neumonia-app

   # borrar la imagen (rebuildear después con: docker build -t uao-neumonia .)
   docker rmi uao-neumonia

   # opcional: limpiar cachés de build colgantes
   docker image prune -f
   ```

## 🏗️ Arquitectura (MVC)

[**Diagrama interactivo de arquitectura**](docs/arquitectura.html) — generado con
[Archify](https://github.com/tt-a1i/archify) a partir del código real
(fuente versionada: [`docs/arquitectura.json`](docs/arquitectura.json)).

```text
src/
├── models/                # capa modelo — aquí vive TODO lo pesado
│   ├── read_img.py        #   DICOM/JPG → (array BGR uint8, preview PIL)
│   ├── preprocess_img.py  #   tensor (1, 512, 512, 1) float32 [0-1] con CLAHE
│   ├── load_model.py      #   carga lazy de TF + validación de WilhemNet_86
│   └── grad_cam.py        #   heatmap Grad-CAM + overlay (explicabilidad)
├── controllers/
│   └── integrator.py      # fachada única con modelo INYECTADO (DI)
└── views/
    └── app.py             # interfaz Streamlit (cáscara fina; solo habla con Integrator)
```

Reglas MVC (detalle en [`CONSTITUTION.md`](CONSTITUTION.md) §5):

- La vista NUNCA importa `tensorflow`/`cv2`/`pydicom`: solo habla con `Integrator`.
- Solo `load_model.py` (y `grad_cam.py` bajo demanda) conocen TensorFlow, con imports
  diferidos (*lazy*).
- El modelo entra por inyección: producción = Keras real; pruebas = `FakeModel`
  (dobles en `test/conftest.py`) — así los 120 tests AAA corren sin TF en local.

## ✅ Calidad

```bash
make help         # lista todos los objetivos
make dev          # entorno + hooks de pre-commit
make lint         # Ruff (lint)
make format       # Ruff (formato)
make verify       # recepción del entorno (7 checks de verify_pipeline.py)
make test         # pytest (9 AAA del iniciador · objetivo del equipo: 120)
make check        # lint + formato + recepción + pruebas — verde ANTES de push
```

- `pre-commit` ejecuta higiene de archivos + Ruff + `uv-lock` en cada commit
  (`--no-verify` está prohibido salvo causa documentada).
- La CI replica `make check`(`.github/workflows/ci.yml`) en cada push/PR hacia
  `main` y `develop`.

## 🧠 Acerca del modelo

La red neuronal convolucional implementada (CNN) está basada en el modelo de
F. Pasa, V. Golkov, F. Pfeifer, D. Cremers y D. Pfeifer en su artículo
*Efficient Deep Network Architectures for Fast Chest X-Ray Tuberculosis
Screening and Visualization*.

Está compuesta por **5 bloques convolucionales**, cada uno con 3 convoluciones: dos
secuenciales y una conexión *skip* que evita el desvanecimiento del gradiente a
medida que se avanza en profundidad. Con 16, 32, 48, 64 y 80 filtros de 3x3 para
cada bloque respectivamente.

Después de cada bloque convolucional hay una capa de max pooling y, después de la
última, una capa de Average Pooling seguida por tres capas *fully-connected*
(Dense) de 1024, 1024 y 3 neuronas respectivamente.

Para regularizar el modelo se usan 3 capas de Dropout al 20%: dos en los bloques
4 y 5 conv, y otra después de la primera capa Dense.

## 🔥 Acerca de Grad-CAM

Es una técnica para resaltar las regiones de la imagen importantes para la
clasificación. Un mapeo de activaciones de clase para una categoría en particular
indica las regiones de imagen relevantes utilizadas por la CNN para identificar
esa categoría.

Grad-CAM calcula el gradiente de la salida correspondiente a la clase a visualizar
con respecto a las neuronas de una capa de la CNN, lo que da la importancia de cada
neurona en el proceso de decisión de esa clase. Con esos pesos se combina linealmente
el mapa de activaciones de la capa y se refleja sobre la imagen de entrada como un
mapa de calor con intensidades más altas en las regiones relevantes.

## 🤝 Contribución

Lee la [guía de contribución](CONTRIBUTING.md) (entorno, Gitflow, estándares y
commits) y el [código de conducta](CODE_OF_CONDUCT.md). Destaca el flujo:

- ramas `feature/<modulo>` → PR hacia `develop` con la
  [plantilla de PR](.github/pull_request_template.md);
- el **code reviewer** — Luis Fernando Meza (`@luisfermezaramrez-stack`) — es la única persona que revisa, aprueba y hace
  merge; **nadie se auto-aprueba**;
- cada integrante reclama su archivo del
  [reparto de tests](test/README.md) (15+30+25+10+15+9+16 = **120**) y
  **sustenta lo que escribe**.

## 📄 Licencia y créditos

Este fork académico se distribuye bajo [Licencia MIT](LICENSE).

- **Iniciador del fork:** Juan Camilo Perez — port de infraestructura (MVC, tests,
  CI, estándar del equipo).
- **Proyecto original:** Isabella Torres Revelo · Nicolas Diaz Salazar
  (taller dalquinones) — modelo, dataset y UI legada.
- Modelo base: F. Pasa et al. — *Efficient Deep Network Architectures…*
