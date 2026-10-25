# CONSTITUTION — UAO-Neumonia

> **Verdad central del proyecto.** Este documento define el propósito, el stack, la
> estructura, las convenciones y el flujo de trabajo de `UAO-Neumonia`. Si algo aquí
> contradice al código, el código manda **y este archivo se actualiza**.

> ⚠️ **Directiva de mantenimiento:** cada vez que se realicen cambios significativos
> (nuevas funcionalidades, cambios arquitectónicos, adición de clases/módulos o ajustes
> de tecnologías), **se debe actualizar este archivo en el mismo commit** para mantener
> la verdad central del proyecto bien documentada.

---

## 0. Identidad del proyecto

| Campo | Valor |
|-------|-------|
| **Nombre** | UAO-Neumonia |
| **Descripción** | Detección rápida de neumonía con CNN sobre radiografías de tórax (DICOM/JPG), con Grad-CAM como explicabilidad, refactorizada a arquitectura MVC |
| **Repositorio** | `git@github.com:JuanC0319/UAO-Neumonia.git` (fork académico del taller original) |
| **Autoría** | Equipo del taller — iniciador del proyecto: Juan Camilo Perez — kamilo199603@gmail.com |
| **Licencia** | [MIT](LICENSE) (fork académico; © 2026 Juan Camilo Perez) |
| **Rama de integración** | `develop` (Gitflow) |
| **Estado** | Refactor MVC completo + estandar aplicado (Makefile, pre-commit, CI, Archify) — vista/Docker/PDF pendientes 🚧 |

## 1. Propósito y no-propósito

### Propósito

Clasificar radiografías de tórax en 3 clases — **bacteriana, normal, viral** — con
explicabilidad Grad-CAM, en un código dividido en MVC, testeable a >95 % sin
TensorFlow local y ejecutable en Mac Intel, Windows, Linux, Docker y CI.

### No-propósito

- NO es un sistema de diagnóstico médico certificado; es un producto académico.
- NO reentrena la red: solo se sirve el checkpoint `WilhemNet_86.h5`.
- NO provee persistencia de pacientes (la CSV/Excel del monolito queda fuera del MVP).

## 2. Alcance

### Dentro del alcance

Migración MVC del monolito `detector_neumonia.py` → `src/models` + `src/controllers`
+ `src/views`; **120 tests pytest** (AAA); Docker CPU basado en `uv`; interfaz gráfica
(Tkinter o Streamlit, según decisión del equipo); CI/CD con GitHub Actions y Gitflow
con PRs y plantilla; reporte PDF final.

### Fuera del alcance (por ahora)

Retraining/fine-tuning; despliegue GPU; migración al layout de paquete Python del
estándar personal (`uv_build`) — el proyecto corre en **modo virtual** (ver §9).

## 3. Stack técnico

| Categoría | Tecnología |
|-----------|------------|
| **Lenguaje** | Python 3.13 (`.python-version` — pin real 3.13.16) |
| **Gestor de paquetes y entorno** | [`uv`](https://docs.astral.sh/uv/) — **único**, `uv.lock` versionado |
| **Testing** | [`pytest`](https://docs.pytest.org/) (`testpaths = ["test"]`, `pythonpath = ["."]`) |
| **Linting / Formato** | [Ruff](https://docs.astral.sh/ruff/) (line-length = 100) |
| **Higiene de commits** | [`pre-commit`](https://pre-commit.com/) (Ruff + higiene de archivos + `uv-lock`) |
| **Interfaz de tareas** | `Makefile` — `make check` (lint + formato + recepción + pruebas) |
| **CI/CD** | GitHub Actions — [`.github/workflows/ci.yml`](.github/workflows/ci.yml) (paridad local↔CI) |
| **DeepLearning** | TensorFlow (SOLO en Docker/CI; lazy en local) |
| **Visión/DICOM** | OpenCV `4.12.0.88` headless · pydicom · Pillow |
| **Diagramas** | [Archify](https://github.com/tt-a1i/archify) — `docs/arquitectura.{json,html}` |

> ⚠️ **Regla dura:** toda gestión de dependencias y entornos pasa por `uv`. Está
> **prohibido** `pip`, `conda` y `requirements.txt`. Dependencias nuevas SOLO con
> `uv add` / `uv add --dev`.

## 4. Estructura del repositorio

```text
UAO-Neumonia/
├── src/models/            # capa modelo: read_img, preprocess_img, load_model, grad_cam
├── src/controllers/       # integrator (fachada MVC con modelo inyectado)
├── src/views/             # interfaz gráfica (en construcción)
├── test/                  # conftest.py (dobles), test_fakes.py (9) y reparto de 120
├── data/                  # fixtures reales committeadas (normal.dcm, radiografia.jpg)
├── docs/arquitectura.{json,html}  # diagrama Archify (JSON = fuente versionada)
├── .github/               # ci.yml · PR template · issue templates · labels · dependabot
├── AGENTS.md              # contexto para asistentes LLM (+ .opencode/rules/tgrep.md)
├── CONSTITUTION.md        # este archivo — verdad central
├── CONVENCIONES.md        # contrato de trabajo del equipo
├── CONTRIBUTING.md + CODE_OF_CONDUCT.md
├── LICENSE                # MIT (fork académico)
├── Makefile + .pre-commit-config.yaml + .editorconfig
├── pyproject.toml + uv.lock
└── verify_pipeline.py     # verificador de recepción (7 checks)
```

## 5. Convenciones de código

- **Estilo:** impuesto por Ruff; docstrings en módulos, clases y funciones públicas.
- **Pruebas:** patrón **AAA** obligatorio; una sola conducta por test; nombres
  autodescriptivos en español.
- **Idioma:** comentarios, docstrings, commits y docs en español; identificadores en
  inglés o español coherente con lo ya escrito.
- **Reglas propias (invariantes del dominio):**
  - **TF-lazy:** solo `src/models/load_model.py` puede conocer TensorFlow, con import
    diferido dentro de la función.
  - **Vista pura:** `src/views` NUNCA importa `tensorflow`/`cv2`/`pydicom`; solo habla
    con `Integrator`.
  - **BGR canónica** en todo el pipeline interno; imágenes PIL solo en la frontera con
    la vista (`preview`).
  - **Contrato de inferencia:** `preprocess` entrega tensor `(1, 512, 512, 1)`
    float32 en [0, 1]; `LABELS = {0:"bacteriana", 1:"normal", 2:"viral"}` según el
    orden alfabético del dataset de entrenamiento.
  - **Modelo inyectado:** `Integrator(model)` con DI — producción: Keras real; pruebas:
    dobles (`FakeModel`/`FakeNet` de `test/conftest.py`).
  - **Artefacto pesado:** `WilhemNet_86.h5` (~112 MB) vive en la raíz pero está en
    `.gitignore` (se comparte por unidad externa, no por git).

## 6. Calidad y automatización

| Herramienta | Rol | Comando |
|-------------|-----|---------|
| `uv` | Entornos y dependencias | `uv sync` |
| `Makefile` | Interfaz de tareas (paridad local↔CI) | `make check` |
| `verify_pipeline.py` | Verificación de recepción (7 checks) | `make verify` / `uv run python verify_pipeline.py` |
| Ruff | Lint y formato | `make lint` / `make format` |
| pre-commit | Higiene + Ruff + `uv-lock` antes de cada commit | `uv run pre-commit run --all-files` |
| pytest | Pruebas (AAA, sin TF local) | `make test` / `uv run pytest -v` |
| GitHub Actions | CI: `ci.yml` (pre-commit + recepción + pytest) | automático en push/PR (`main`, `develop`) |
| Archify | Diagrama de arquitectura desde código | `docs/arquitectura.{json,html}` |

Nada se sube con la verificación en rojo; se prohíbe saltarse verificaciones sin causa
documentada en el propio commit (`--no-verify` de pre-commit incluido). `make check`
es la puerta de entrada; la CI replica esa misma paridad en GitHub.

## 7. Repositorio y Git

- **Remoto:** `git@github.com:JuanC0319/UAO-Neumonia.git` (fork de `dalquinones/UAO-Neumonia`).
- **Gitflow:** `feature/…` → PR (plantilla obligatoria) → `develop` → `main` al cierre.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) en español e
  imperativo: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- **Gobernanza GitHub:** plantilla de PR con checklist; issue templates (bug,
  funcionalidad, spike); labels sincronizadas vía `.github/labels.yml`; Dependabot
  (`pyproject` + Actions); protección de ramas al cerrar hacia `main`.
- **Identidad git:** ver §11 (configuración local de máquina; nunca versionada).

## 8. Flujo de trabajo del equipo

**Roles separados — quien coordina NO es quien revisa:**

- **Iniciador del proyecto** (dueño del fork `JuanC0319`): porta la infraestructura
  compartida (`test/conftest.py`, `test/README.md`, `verify_pipeline.py`), coordina el
  reparto y los pasos del taller. **NO revisa PRs.**
- **Code Reviewer — Luis Fernando Meza (`@luisfermezaramrez-stack`)**: única persona
  que revisa, aprueba y hace merge de los PRs hacia `feature/refactor-mvc`, y después
  hacia `develop`.
- **Cada integrante:** reclama su archivo del reparto de `test/README.md`
  (15+30+25+10+15+9+16 = 120), escribe sus pruebas y **sustenta lo que escribe**.
- Antes de pedir review: `uv run python verify_pipeline.py` →
  `uv run pytest test/test_<módulo>.py` → `uv run ruff check .` — los tres en verde.

## 9. Historial de decisiones técnicas

| Fecha | Decisión | Motivo |
|-------|----------|--------|
| 2026-10-19 | `uv` como único gestor; Python 3.13; pins exactos (`opencv==4.12.0.88`, `numpy<2.3`) | Mac Intel x86_64 macOS 13.7: OpenCV 5.x y TF no tienen ruedas aplicables localmente |
| 2026-10-19 | Proyecto **virtual** (sin paquete ni build-system; imports `src.*` vía `pythonpath`) | App de taller; en línea con el estándar personal (se migraría a layout de paquete si se decidiera) |
| 2026-10-20 | Refactor MVC: módulos modelo + controlador `Integrator` con DI | Sustentabilidad, testabilidad y separación de responsabilidades |
| 2026-10-20 | Patrón TF-lazy + `FakeModel`/`FakeNet` | 120 tests en verde sin TF local; TF real solo en Docker/CI |
| 2026-10-20 | Gitflow con PRs y plantilla; `CONVENCIONES.md` | La rúbrica exige colaboración gobernada |
| 2026-10-21 | `CONSTITUTION.md` como fuente única de verdad; retiro definitivo de `requirements.txt` | Estándar del equipo; fin de conda/pip |
| 2026-10-24 | Estandar `jan-project-standard` aplicado: `Makefile`, pre-commit, `.editorconfig`, LICENSE MIT, CONTRIBUTING/CODE_OF_CONDUCT/AGENTS, README profesional con badges | Calidad automatizada (`make check` verde antes de push); puerta de entrada profesional al fork |
| 2026-10-24 | CI `ci.yml` (pre-commit + recepción + pytest) y gobernanza GitHub (issue templates, labels, Dependabot) | Paridad local↔CI exigida por el estándar; la rúbrica pide CI/CD gobernado |
| 2026-10-24 | Diagrama de arquitectura con Archify: `docs/arquitectura.json` (fuente) + `docs/arquitectura.html` | Documentación viva del MVC, respaldada por líneas del código (`meta.repository` pin `bf3eaf5`) |
| 2026-10-24 | Code reviewer designado: **Luis Fernando Meza** (`@luisfermezaramrez-stack`) | Cierra el rol pendiente de §8; los PRs se dirigen con @mention real (notificaciones de GitHub) |

## 10. Roadmap y tareas pendientes

- [x] Refactor MVC: modelos + controlador (5 commits `feat:`)
- [x] Infra de colaboración: CONVENCIONES, PR template, verificador, test/README
- [x] **Paso 3 (base):** `pythonpath = ["."]` + `test/conftest.py` + `test/test_fakes.py`
      (9 pruebas AAA del iniciador) — pytest en verde (9 passed, 2026-10-24)
- [ ] **Paso 3 (equipo):** reparto de los 111 tests restantes vía PRs (claims en tabla)
- [ ] **Paso 3b:** decidir vista (Tkinter vs Streamlit) y construirla con hook
      `Integrator.explain()`
- [x] **Cierre del refactor:** `git rm detector_neumonia.py` + README migrado a la
      arquitectura MVC + `requirements.txt` retirado (2026-10-24)
- [x] **Estandar de proyecto aplicado:** Makefile + pre-commit + LICENSE MIT +
      README profesional (badges, TOC) + diagrama Archify + gobernanza GitHub
      (`ci.yml`, issue templates, labels, Dependabot) (2026-10-24)
- [ ] **Paso 4:** Dockerfile `python:3.13-slim` + `uv` (CPU, `MODEL_PATH`) + `.dockerignore`
- [ ] **Paso 5:** aplicar CI a la rama (merge `feature` → `develop` con aprobación del
      reviewer; el `ci.yml` ya existe) + protección de `main`
- [ ] **Paso 6:** reporte PDF (diagrama MVC, evidencias, decisiones)

## 11. Configuración local (no trackear)

> Configuración específica de cada máquina. **Nunca se commitea.**

La identidad git actual de la máquina se resuelve por el `git config --global`
(`git config user.name/user.email`). Para **identidad automática por carpeta** se
adopta el estándar de la skill `jan-project-standard` con `includeIf` (Perfil B):
bloque en `~/.gitconfig` apuntando a `~/.gitconfig-<perfil>` con `user.name`,
`user.email` y `sshCommand`. Guía completa: skill `jan-project-standard` →
`references/git-identity.md` (macOS/Windows, `gitdir/i:`, `IdentitiesOnly`).

```ini
# ~/.gitconfig  (ejemplo — datos propios de cada integrante)
[includeIf "gitdir:~/ruta/del/os-proyectos/"]
    path = ~/.gitconfig-personal
```
