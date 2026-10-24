# AGENTS — UAO-Neumonia

> Contexto del proyecto para sesiones con asistentes LLM (OpenCode, Antigravity,
> GitHub Copilot CLI, Claude Code, etc.). Trackeado al repositorio.

## Proyecto

Detector de neumonia con CNN y Grad-CAM - taller MVC con uv y Python 3.13.

> 📘 La verdad técnica central —propósito, stack, estructura, convenciones, decisiones y
> flujo de trabajo— vive en [`CONSTITUTION.md`](./CONSTITUTION.md). **Léela antes de
> operar.** Ante cualquier cambio arquitectónico, tecnológico o de proceso, actualízala
> en el mismo commit.

---

## Flujo Diario (commit + push)

```bash
# 1. Ver cambios
git status

# 2. Verificación completa (lint + formato + pruebas)
make check

# 3. Agregar y commitear (los hooks de pre-commit se ejecutan solos)
git add -A
git commit -m "tipo: mensaje descriptivo"

# 4. Subir a GitHub
git push
```

**Convención de commits:** [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — nueva funcionalidad
- `fix:` — corrección de bug
- `docs:` — documentación
- `refactor:` — refactorización
- `style:` — formato, linting (sin cambio de lógica)
- `test:` — pruebas
- `chore:` — mantenimiento o configuración

---

## Comandos del proyecto (`make`)

```bash
make help      # lista los objetivos disponibles
make dev       # entornos + hooks de pre-commit
make check     # lint + formato + recepción (verify_pipeline) + pruebas
make lint      # Ruff (lint)
make format    # Ruff (formato)
make verify    # verificador de recepción del entorno
make test      # pytest
```

> ⚠️ **Regla dura:** toda gestión de dependencias y entornos pasa por `uv`. Está
> **prohibido** `pip` y `requirements.txt`.

---

## Búsqueda de código: `tgrep`

Si [`tgrep`](https://github.com/microsoft/tgrep) está instalado, úsalo para búsquedas de
código en lugar de `grep` o `find` (índice de trigramas en `.tgrep/`, ignorado por Git):

```bash
tgrep -- "patrón" .            # búsqueda regex (usa -- antes del patrón)
tgrep -F -- "simbolo" .        # coincidencia literal
tgrep -t py -- "def " .        # filtrar por tipo de archivo
tgrep -g "src/**" -- "kw" .    # filtrar por ruta
tgrep -l -- "import torch" .   # listar archivos coincidentes
tgrep index .                  # (re)indexar
```

En OpenCode la preferencia se aplica automáticamente mediante la regla
[`.opencode/rules/tgrep.md`](./.opencode/rules/tgrep.md).

---

## Estructura y convenciones

- `src/models` + `src/controllers` + `src/views` — arquitectura MVC del taller
  (detalle en `CONSTITUTION.md` §4).
- `test/` — pytest con dobles y fixtures (`conftest.py`; `testpaths=["test"]` en pyproject).
- `verify_pipeline.py` — verificador de recepción del entorno (7 checks).
- `make check` = lint + formato + recepción + pruebas; **verde antes de cada push**.
- `docs/arquitectura.{json,html}` — diagrama de arquitectura con Archify (el JSON es la
  fuente versionada; el HTML se regenera con la skill, nunca a mano).
- `CONSTITUTION.md` — fuente única de verdad.
- Estilo impuesto por **Ruff** (`line-length = 100`, comillas dobles).
- Idioma: UI, comentarios y commits en español; identificadores en inglés.
- Gitflow: ramas `feature/<modulo>` → PR hacia `develop`; la revisión y el merge los hace
  solo el **code reviewer** designado (el iniciador no se auto-aprueba).
