# 🤝 Guía de contribución

¡Gracias por tu interés en contribuir a **UAO-Neumonia**!

La documentación central del proyecto vive en [`CONSTITUTION.md`](./CONSTITUTION.md).

> 📘 **Antes de tu primera contribución**, lee [`CONSTITUTION.md`](./CONSTITUTION.md)
> (stack, decisiones y convenciones) y el [Código de conducta](./CODE_OF_CONDUCT.md).

---

## 📑 Tabla de contenidos

- [Requisitos previos](#-requisitos-previos)
- [Configuración del entorno](#-configuración-del-entorno)
- [Flujo de trabajo con Git](#-flujo-de-trabajo-con-git)
- [Estándares de código](#-estándares-de-código)
- [Mensajes de commit](#-mensajes-de-commit)
- [Pull Requests](#-pull-requests)
- [Reportar issues](#-reportar-issues)
- [Regla de oro: mantener la constitución](#-regla-de-oro-mantener-la-constitución)

---

## ✅ Requisitos previos

| Herramienta | Versión | Notas |
|-------------|---------|-------|
| [Python](https://www.python.org/downloads/) | 3.13+ | Fijada en `.python-version` |
| [`uv`](https://docs.astral.sh/uv/getting-started/installation/) | Última estable | Gestor de paquetes y entornos |
| [Git](https://git-scm.com/) | Última estable | Control de versiones |
| [GitHub CLI](https://cli.github.com/) | Opcional | Facilita issues y PRs |

> ⚠️ **Regla del proyecto:** está **prohibido** usar `pip` o `requirements.txt`.
> Toda gestión de dependencias pasa por `uv`.

## ⚙️ Configuración del entorno

```bash
git clone https://github.com/JuanC0319/UAO-Neumonia.git
cd UAO-Neumonia

uv sync
uv run pre-commit install   # equivalente a: make dev
uv run python verify_pipeline.py   #recepción del entorno (7 checks)
make check
```

## 🌿 Flujo de trabajo con Git

El proyecto usa **Gitflow académico** con dos ramas permanentes y roles separados:

| Rama | Rol |
|------|-----|
| `main` | Estable y **protegida**; se actualiza solo con merges revisados |
| `develop` | Rama de integración del taller (destino de los PRs) |
| `feature/<modulo>` | Rama de trabajo de cada módulo |

1. Actualiza `develop`:
   ```bash
   git checkout develop && git pull origin develop
   ```
2. Crea una rama descriptiva:
   ```bash
   git checkout -b feat/nombre-corto
   ```
   | Prefijo | Uso |
   |---------|-----|
   | `feat/` | Nueva funcionalidad |
   | `fix/` | Corrección de bug |
   | `docs/` | Documentación |
   | `refactor/` | Refactorización |
   | `test/` | Pruebas |
   | `chore/` | Mantenimiento o configuración |
3. Desarrolla en commits pequeños y atómicos.
4. Abre un Pull Request hacia `develop` (nunca hacia `main`).

> 💡 **Regla de oro:** nunca hagas `push` "para ver si funciona". Verifica localmente
> con `make check` antes de abrir el PR; el code reviewer (`@luisfermezaramrez-stack`)
> aprueba y hace el merge.

## 🧹 Estándares de código

- **Estilo:** [PEP 8](https://peps.python.org/pep-0008/) y [PEP 257](https://peps.python.org/pep-0257/).
- **Tipado:** anotaciones en funciones y firmas públicas.
- **Nomenclatura:** `snake_case`, `PascalCase` (clases), `UPPER_CASE` (constantes).
- **Pruebas:** `pytest` con patrón **AAA** (*Arrange · Act · Assert*).
- **Lint y formato:** Ruff.

```bash
make lint     # uv run ruff check .
make format   # uv run ruff format .
make test     # uv run pytest
```

> ❌ **Nunca** subas secretos (`.env`, tokens, llaves), datos pesados ni entornos
> virtuales. Respeta el [`.gitignore`](./.gitignore).

## 📝 Mensajes de commit

Se usa la especificación de [Conventional Commits](https://www.conventionalcommits.org/):

```text
<tipo>(<ámbito opcional>): <descripción breve en imperativo>
```

| Tipo | Uso |
|------|-----|
| `feat` | Nueva funcionalidad |
| `fix` | Corrección de bug |
| `docs` | Documentación |
| `refactor` | Refactorización sin cambio de comportamiento |
| `style` | Formato, linting (sin cambio de lógica) |
| `test` | Pruebas |
| `chore` | Mantenimiento o configuración |

## 🔀 Pull Requests

1. Verifica que tu rama esté actualizada respecto a `develop`.
2. Abre el PR y completa la plantilla oficial.
3. Usa un título con prefijo convencional y mantén el PR **acotado a un solo propósito**.
4. Marca los checklist aplicables (pruebas, lint, `uv.lock`, `CONSTITUTION.md`, etc.).
5. La integración se realiza por **Squash & Merge** (salvo indicación contraria).

## 🐞 Reportar issues

Usa las plantillas disponibles en `.github/ISSUE_TEMPLATE/` (bug, funcionalidad, spike)
e incluye contexto, pasos para reproducir y evidencia cuando aplique.

## 🏛️ Regla de oro: mantener la constitución

> Cualquier cambio **arquitectónico, tecnológico o de proceso** debe reflejarse en
> [`CONSTITUTION.md`](./CONSTITUTION.md). El checklist del PR lo verifica de forma explícita.

---

## 📄 Licencia

Al contribuir, aceptas que tus aportes se distribuyan bajo la [Licencia MIT](./LICENSE).
