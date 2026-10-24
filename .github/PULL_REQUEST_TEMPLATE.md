# PR de <nombre del módulo/cambio>

## Módulo afectado
<!-- marca con X lo que aplica -->
- [ ] read_img (`src/models/read_img.py`)
- [ ] preprocess_img (`src/models/preprocess_img.py`)
- [ ] load_model (`src/models/load_model.py`)
- [ ] grad_cam (`src/models/grad_cam.py`)
- [ ] integrator / controller (`src/controllers/integrator.py`)
- [ ] view (`src/views/`)
- [ ] infra/tests/docs

## ¿Qué hace este PR?

- (resumen en 2–3 líneas)

## Justificación de diseño (por qué así, no de otra forma)

- Alta cohesión: (responsabilidad única del módulo)
- Bajo acoplamiento: (de qué NUNCA depende este módulo)

## Checklist DoD M1

- [ ] Responsabilidad única y ubicación correcta (models/controllers/views)
- [ ] Type hints + docstrings estilo Google (Args/Returns/Raises)
- [ ] `uv run python -m py_compile <rutas>` sin errores
- [ ] Pruebas AAA del módulo en verde (`uv run pytest`), cuando aplique
- [ ] Commit(s) con prefijo convencional (feat/fix/test/docs/refactor/chore)
- [ ] La vista NO importa tensorflow/cv2/pydicom (si aplica)
- [ ] Sin binarios grandes rastreados (.h5 ignorado)

## Evidencia

```
(paste aquí la salida de tus pruebas/verificaciones)
```

## Checklist del estándar (jan-project-standard)

- [ ] Si el cambio es arquitectónico/técnico/de proceso: `CONSTITUTION.md` actualizado **en el mismo commit**
- [ ] `uv.lock` sincronizado vía `uv add`/`uv sync` (nunca editado a mano)

## ¿A quién avisar?

- **Code reviewer:** @<code-reviewer> — única persona que revisa, aprueba y hace merge
- **Iniciador (dueño del fork):** @JuanC0319
