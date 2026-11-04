# Proyecto Neumonía — refactor MVC (UAO, Módulo 1)

> Proyecto académico basado en [UAO-Neumonia de dalquinones](https://github.com/dalquinones/UAO-Neumonia).
> **No utilizar para diagnóstico ni decisiones clínicas.**

## Qué hace

- Permite **cargar y mostrar** radiografías DICOM (`.dcm`), JPEG (`.jpg`, `.jpeg`) o PNG.
- Ejecuta una clasificación experimental: **neumonía bacteriana / viral / normal**.
- Muestra las puntuaciones softmax y, cuando es posible, un mapa **Grad-CAM**.
- Se organiza en los seis módulos de la **Clase 1** (diapositiva 27) y aplica
  Clean Code, SOLID, TDD, pytest y el flujo de Git de la **Clase 2**.

## Diferencia frente al sistema original

El repositorio base no incluye el archivo `.h5` de pesos que cita el proyecto.
Esta implementación integra **un modelo entrenado distinto**, alojado en
[kiselyovd/chest-xray-classifier](https://huggingface.co/kiselyovd/chest-xray-classifier)
(ConvNeXtV2; la revisión está fijada en `load_model.py`).

La primera predicción descarga pesos y requiere internet. Se usa el
preprocesamiento académico 512×512/gris/CLAHE/[0,1] y una **adaptación adicional**
a las entradas 224×224 RGB del modelo alternativo. **La precisión de este
modelo con la adaptación no ha sido validada**, las puntuaciones no son riesgos
clínicos calibrados y no se afirma reproducir el `.h5` original.

## Requisitos

- Windows (PowerShell) o Linux/macOS.
- Python 3.13 gestionado por [`uv`](https://docs.astral.sh/uv/).
- Git; opcionalmente `make` y Docker Desktop.
- Internet para **instalar dependencias** y la primera descarga del modelo.

### Iniciar (Windows PowerShell)

```powershell
# Dentro del fork que creaste en GitHub y clonaste localmente:
uv sync                 # Genera uv.lock cuando aún no existe
uv run pytest -q        # Pruebas unitarias sin descargar el modelo
uv run ruff check --fix .
uv run ruff format .
uv run ruff check .
uv run ruff format --check .
uv run pytest -W error -q
uv run streamlit run app.py
```

La aplicación abre normalmente en `http://localhost:8501`. El modelo puede
necesitar una descarga grande y bastante memoria RAM.

**IMPORTANTE: confirmar que el archivo `uv.lock` generado quedó versionado en Git.**
Si el fork conserva un `requirements.txt` antiguo del monolito, retirarlo del
repositorio (`git rm requirements.txt`) para usar únicamente `uv`, como pide la clase.

### Con Make (instalado en el equipo)

```bash
make install
make run
make check
```

### Con Docker

```bash
docker build -t uao-neumonia .
docker run --rm -p 8501:8501 uao-neumonia
```

Abrir `http://localhost:8501`. Comprobar el build real antes de entregar;
las pruebas unitarias no verifican la imagen de Docker.

## Organización de carpetas

```text
UAO-Neumonia/                  ← fork del proyecto original
├── detector_neumonia.py       ← código original, conservar en el fork
├── app.py                     ← entrada Streamlit
├── src/neumonia/
│   ├── read_img.py            ← DICOM y JPEG/PNG
│   ├── preprocess_img.py      ← 512×512, gris, CLAHE, [0,1]
│   ├── load_model.py          ← carga del modelo 1 vez
│   ├── grad_cam.py            ← mapa de atribución
│   ├── integrator.py          ← controlador y coordinación
│   └── ui.py                  ← vista
├── tests/                     ← pruebas con pytest
├── docs/ARQUITECTURA.md       ← diagrama, contratos y decisiones
├── docs/arquitectura.png      ← imagen requerida en el aula
├── pyproject.toml
├── uv.lock                    ← generar con uv sync ANTES de entregar
├── .python-version
├── .pre-commit-config.yaml
├── .github/pull_request_template.md
├── Makefile
├── Dockerfile
└── .gitignore
```

## Diseño y decisiones

El patrón es MVC: `ui.py` (Vista) → `integrator.py` (Controlador) →
`read_img.py`, `preprocess_img.py`, `load_model.py`, `grad_cam.py` (Modelo).
La Vista no importa TensorFlow/PyTorch y no abre archivos en disco.
El controlador acepta un motor de inferencia inyectado para sustituirlo en tests.
Ver [documentación completa](docs/ARQUITECTURA.md) y [diagrama](docs/arquitectura.png).

## Pruebas y estilo

Las pruebas de `tests/` cubren lectura, archivos inválidos, forma, rango y tipo
512×512, interpretación de probabilidades, controlador y superposición visual.

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run pytest -W error -q
```

Se incluye `.pre-commit-config.yaml`. Instalar los hooks si el entorno dispone
de `pre-commit`:

```bash
uv run pre-commit install
```

## Entrega con Gitflow ligero (Clase 2)

La presentación recomienda trabajar mediante `feature/*` y un Pull Request,
**sin subir cambios directamente a `main`**:

```bash
git switch -c feature/refactor-neumonia
# Copiar los archivos del ZIP dentro del fork (sin borrar el .git original).
uv sync
git add .
git commit -m "refactor: separa pipeline de neumonia en seis modulos"
git push -u origin feature/refactor-neumonia
```

En GitHub, usar **Compare & pull request**, completar la plantilla, pedir
revisión a un compañero y fusionar después de verificar `make check`.
Finalmente pegar en el aula el **enlace al repositorio propio** e indicar los
nombres reales de quienes participaron.

## Límites de la validación

Las pruebas sintéticas no verifican diagnóstico, precisión del modelo, ni que
el contenedor cargue correctamente los pesos. Esa es una prueba de integración
pendiente en el equipo del estudiante con conectividad y memoria suficientes.

**Integrantes reales:** [agregar nombres de estudiantes participantes].
