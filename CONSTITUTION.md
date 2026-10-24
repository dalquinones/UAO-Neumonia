# CONSTITUCIÓN DEL PROYECTO: UAO-Neumonía

> Contexto del proyecto: Sistema de apoyo al diagnóstico de neumonía usando Inteligencia Artificial (Redes Convolucionales) y Explicabilidad (Grad-CAM).

---

## 📌 Principios de Arquitectura

1. **Clean Architecture (Patrón Fachada):**
   - El sistema se divide en capas estrictas: Presentación (UI), Orquestación (Fachada) y Lógica de Negocio (Modelos/Procesamiento).
   - `src/integrator.py` actúa como la única Fachada. La interfaz gráfica (`detector_neumonia.py`) nunca debe importar módulos de IA directamente.

2. **Alta Cohesión y Bajo Acoplamiento:**
   - Cada archivo tiene una única responsabilidad (Single Responsibility Principle).
   - `read_img.py`: Solo lee y valida.
   - `preprocess_img.py`: Solo transforma y normaliza (CLAHE, 512x512).
   - `load_model.py`: Solo carga y valida la topología del modelo.
   - `grad_cam.py`: Solo infiere y genera mapas de calor.

---

## 🛠️ Stack Técnico Oficial

| Categoría | Tecnología |
|-----------|------------|
| **Lenguaje** | Python 3.13 |
| **Gestor de Entorno** | `uv` (Prohibido usar `pip` directamente o `requirements.txt`) |
| **Interfaz Gráfica** | `tkinter` (con widgets `ttk` para estilos modernos) |
| **Deep Learning** | `tensorflow` (Keras 3) |
| **Visión Artificial** | `opencv-python`, `Pillow` |
| **Testing** | `pytest` |
| **Archivos Médicos** | `pydicom` |

---

## ✅ Reglas de Calidad y Git

- **No rastrear modelos:** Los archivos `.h5` (como `conv_MLP_84.h5`) están estrictamente prohibidos en el historial de Git debido a su tamaño. Deben estar en `.gitignore`.
- **Commits Convencionales:** Todo commit debe usar prefijos (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`).
- **Pruebas Unitarias:** Se exige mantener la cobertura de pruebas. Actualmente contamos con 126 pruebas ejecutables mediante `uv run pytest`.
- **Type Hints y Docstrings:** Todo código nuevo en `src/` debe incluir Type Hints compatibles con PEP8 y Docstrings estilo Google explicando `Args:`, `Returns:` y `Raises:`.

---

## 🚀 Flujo de Desarrollo (Daily Workflow)

```bash
# 1. Instalar dependencias o sincronizar entorno
uv sync

# 2. Correr pruebas antes de enviar cambios
make test  # o uv run pytest -v

# 3. Ejecutar la aplicación para validar GUI
make run   # o uv run python src/detector_neumonia.py
```
