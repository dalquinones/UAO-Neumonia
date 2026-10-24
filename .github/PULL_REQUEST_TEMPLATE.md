<!--
📌 INSTRUCCIONES PARA ESTUDIANTES UAO:
Copia este archivo en la ubicación .github/PULL_REQUEST_TEMPLATE.md en la raíz de tu repositorio de GitHub.
Al crear una Pull Request, GitHub utilizará automáticamente este formato para estructurar las revisiones de código.
-->

## 📌 Título de la Pull Request
Aplica la convención de [Conventional Commits](https://www.conventionalcommits.org/) para clasificar tu trabajo:

| Prefijo | Cuándo usar | Ejemplo de Título |
| :--- | :--- | :--- |
| `feat:` | Nueva funcionalidad o módulo | `feat: agregar preprocesamiento CLAHE en preprocess_img.py` |
| `fix:` | Corrección de errores o eliminación de warnings | `fix: corregir advertencia de deprecación en TensorFlow` |
| `test:` | Adición o refactorización de pruebas unitarias | `test: agregar 15 pruebas unitarias para load_model.py` |
| `refactor:` | Mejoras de estructura (Clean Code) sin cambiar lógica | `refactor: desacoplar la carga de pesos de la inferencia` |
| `chore:` | Tareas de configuración, Docker, Makefile o deps | `chore: actualizar Dockerfile y lockfile de UV` |
| `docs:` | Cambios en documentación o README | `docs: actualizar diagrama de arquitectura en el README` |

---

## 🧱 Módulo o Componente Afectado
Selecciona los componentes en los que trabaja esta PR:
- [ ] **Lectura de Imágenes** (`src/read_img.py`)
- [ ] **Preprocesamiento** (`src/preprocess_img.py`)
- [ ] **Carga del Modelo** (`src/load_model.py`)
- [ ] **Explicabilidad / Grad-CAM** (`src/grad_cam.py`)
- [ ] **Integrador & CLI / UI** (`src/integrator.py`, `src/detector_neumonia.py`)
- [ ] **Pruebas Unitarias** (`test/`)
- [ ] **Infraestructura & Contenedores** (`Dockerfile`, `.gitignore`, `Makefile`, `pyproject.toml`, `uv.lock`)
- [ ] **Documentación & Licencia** (`README.md`, `LICENSE`)

---

## 📚 Descripción de los Cambios

**Módulo / Historia de Usuario Asociada:**
*(Ej. Implementación de mapas de calor Grad-CAM o Redimensionamiento 512x512)*

### 📝 Resumen Técnico:
-

### 📐 Justificación de Diseño & Clean Code (Robert C. Martin):
- **Alta Cohesión:** <!-- Explica brevemente la responsabilidad única del módulo modificado -->
- **Bajo Acoplamiento:** <!-- Explica cómo se evita la dependencia rígida entre componentes -->

---

## ✅ Lista de Chequeo Pre-PR (Estándares del Curso UAO)
Antes de solicitar revisión a tu equipo o docente, verifica que se cumplan las siguientes reglas:

- [ ] **Entorno Virtual de Ejecución:** Probado y ejecutado exitosamente con **`uv`** en **Python 3.13**. *(Prohibido usar `pip` directo o `requirements.txt`)*.
- [ ] **Sin Warnings:** La ejecución de `uv run` corre completamente limpia **sin advertencias ni warnings**.
- [ ] **Control de Exclusiones (`.gitignore`):** Se verificó que el modelo pesado (`conv_MLP_84.h5`) **NO** esté rastreado en Git.
- [ ] **Estructura del Proyecto:** Todo el código fuente está ubicado dentro de `src/` y las pruebas dentro de `test/`.
- [ ] **Pruebas Unitarias (`pytest`):** Se ejecutó `uv run pytest` y todas las pruebas pasaron exitosamente.
- [ ] **Principios Clean Code:** Funciones con responsabilidad única, nombres descriptivos y encapsulación adecuada.

---

## 🧪 Evidencia de Pruebas Ejecutadas
*(Adjunta capturas de pantalla o pega la salida del comando `uv run pytest` demostrando que el código funciona y las pruebas pasaron)*

```bash
# Salida de la suite de pruebas unitarias ejecutadas con UV:
uv run pytest -v
```
