# Guía de pruebas del equipo — Paso 3 · 120 tests AAA

> Objetivo: **120 pruebas pytest en verde**, patrón AAA obligatorio, **sin TensorFlow
> en las máquinas locales** — el modelo se mockea con `FakeModel` (en `conftest.py`)
> y la red real solo se ejercita en CI/Docker.

## El patrón AAA (obligatorio en todas las pruebas)

Tres bloques, siempre en este orden, separados con comentario:

```python
def test_read_dicom_devuelve_tres_canales():
    # Arrange — preparar: entradas y estado
    ruta = "data/normal.dcm"

    # Act — actuar: UNA sola acción
    array, _ = read_dicom(ruta)

    # Assert — afirmar: lo que DEBE cumplirse
    assert array.shape[2] == 3
    assert array.dtype.name == "uint8"
```

Reglas:

1. **Un solo comportamiento por test** (no encadenar asserts de temas distintos).
2. **Nombre autodescriptivo**: quien lee el nombre entiende qué garantiza.
3. **Cero `import tensorflow`** en los tests: el modelo entra por fixture (FakeModel).
4. Errores se prueban con `pytest.raises(NotFoundFile…, match="…")` — normales: ValueError/FileNotFoundError.

## Reparto de los 120 tests (archivo ↔ módulo ↔ responsable)

| Archivo | Módulo | ~Tests | Responsable |
|---|---|---|---|
| `test/test_read_img.py` | read_img | 15 | (reclamar) |
| `test/test_preprocess_img.py` | preprocess_img | 30 | (reclamar) |
| `test/test_integrator.py` | integrator | 25 | (reclamar) |
| `test/test_load_model.py` | load_model | 10 | (reclamar) |
| `test/test_grad_cam.py` | grad_cam | 15 | (reclamar) |
| `test/test_fakes.py` | infraestructura (conftest) | 9 | Iniciador |
| `test/test_pipeline_e2e.py` | pipeline completo | 16 | (reclamar) |
| **Total** | | **120** | |

**Idea clave por archivo** (qué cubrir):

- `test_read_img.py` — shapes/dtypes, preview es PIL, FileNotFoundError en ambos lectores, `read_image` con archivo corrupto (cv2 devuelve None → raise), normalización DICOM (píxel > 8 bits → 0–255).
- `test_preprocess_img.py` — la forma exacta del tensor, dtype, rango 0–1, imagen plana → salida plana (constante), distintos tamaños de entrada → siempre (1,512,512,1), errores: vacío, None, 4 canales, 2 canales aceptados (gris).
- `test_integrator.py` — con FakeModel: label correcto por argmax (probar las 3 clases), probabilities como floats, dict exacto de claves, dispatch por extensión (.dcm vs .jpg), FileNotFoundError, llamado al modelo EXACTAMENTE 1 vez (spy), preview incluida.
- `test_load_model.py` / `test_grad_cam.py` — lógica sin TF real: rutas inexistentes → error, validación de nombre de capa del modelado en una función que recibe el “modelo” con stubs (la carga real con TF queda para Docker/CI con `pytest.importorskip("tensorflow")`).
- `test_pipeline_e2e.py` — el flujo completo de punta a punta con FakeModel inyectado: varios archivos reales de `data/`, resultados repetibles, contrato del dict, “no explota” con clases empatacas.

## Flujo de trabajo de cada responsable

```bash
uv run python verify_pipeline.py          # 1) entorno sano
uv run pytest test/test_<tu_módulo>.py    # 2) tus pruebas en verde
uv run ruff check .                        # 3) estilo limpio
```

Después: PR hacia `feature/refactor-mvc` con la plantilla (incluye sección de evidencia).

> La pieza compartida `conftest.py` ya existe en esta carpeta: léela antes de
> escribir tus pruebas — tus fixtures (FakeModel, FakeNet, rutas de data/)
> están listos para inyectarse por nombre.
