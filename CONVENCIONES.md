# Convenciones del equipo — M1 · refactor MVC

> Regla de oro: **cada módulo tiene UN responsable**, y ese responsable debe poder
> explicar cada línea de su módulo en la sustentación. Quien escribe, defiende.

## Estructura MVC (no negociable)

| Carpeta | Puede contener | PROHIBIDO |
|---|---|---|
| `src/models/` | `tensorflow`, `cv2`, `pydicom`, `numpy`, `PIL` | imports de UI (Tkinter/Streamlit) |
| `src/controllers/integrator.py` | fachada única; recibe el modelo **inyectado** | lógica dewidgets o de parsing DICOM |
| `src/views/` | Tkinter/Streamlit, pintar imágenes | importar `tensorflow`/`cv2`/`pydicom` |

## Reglas de código (aplican a todos)

- Type hints + docstrings estilo Google (`Args / Returns / Raises`).
- Números mágicos **fuera**: constantes nombradas arriba del módulo (ej. `TARGET_SIZE`).
- Errores explícitos (`FileNotFoundError`, `ValueError` con mensaje claro) — nada silencioso.
- Todos los módulos se pueden probar **sin TensorFlow** (TF se importa dentro de funciones; el modelo se mockea).
- Antes de pushear: `uv run python -m py_compile <tu_archivo>` sin errores.

## Git (flujo)

1. Commits convencionales: `feat | fix | test | docs | refactor | chore`.
2. Cada módulo en su rama: `feature/<nombre-modulo>`.
3. PR hacia `feature/refactor-mvc` usando la plantilla (`.github/PULL_REQUEST_TEMPLATE.md`).
4. Revisión: Juan revisa y hace merge (o pide cambios en el PR).
5. Al cerrar todos los módulos: un único PR `feature/refactor-mvc` → `develop`.
6. Después de cada `git pull`: correr `uv sync` (el `uv.lock` manda; nunca pip).

## Reparto de trabajo

| # | Módulo | Archivo | Responsable |
|---|---|---|---|
| 1 | read_img ✅ | `src/models/read_img.py` | Juan |
| 2 | preprocess_img | `src/models/preprocess_img.py` | (asignar) |
| 3 | load_model | `src/models/load_model.py` | (asignar) |
| 4 | grad_cam | `src/models/grad_cam.py` | (asignar) |
| 5 | integrator (controller) | `src/controllers/integrator.py` | (asignar) |
| 6 | view + decisión Tkinter/Streamlit | `src/views/` | (asignar) |

Cada responsable escribe también las pruebas AAA de su módulo (`test/test_<modulo>.py`, Paso 3).
