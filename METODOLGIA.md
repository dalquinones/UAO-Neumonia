# 📑 Bitácora Técnica de Ingeniería, Errores y Decisiones Arquitectónicas (ADR)
## Proyecto UAO-Neumonía • Especialización en Inteligencia Artificial
**Universidad Autónoma de Occidente (UAO)**  
**Autor Rama:** Miguel Ángel Ortiz
**Fecha:** Septiembre 2026  
**Versión del Documento:** 2.0 (Edición Definitiva y Modular)

---

## Resumen Ejecutivo

El presente documento registra de forma exhaustiva el proceso de modernización, refactorización arquitectónica y aseguramiento de calidad del software para el proyecto **UAO-Neumonía**. 

El repositorio original fue concebido como un prototipo monolítico (*script espagueti*) diseñado para **Python 3.8, TensorFlow 1.x/2.8, Pillow 9, pydicom 2.3 y Windows**. El curso exige desarrollar estrictamente en **Python 3.13** utilizando el gestor moderno **UV**, lo que implicó saltar a versiones contemporáneas (**TensorFlow 2.21, Keras 3.15, Pillow 12, pydicom 3.0 y NumPy 2.5**).

La gran mayoría de fallas encontradas no se debían a deficiencias en el modelo de Machine Learning, sino a **rupturas de compatibilidad de API (Breaking Changes), incompatibilidades gráficas en servidores modernos de pantalla (Wayland en Linux), fugas de rendimiento por recarga redundante de modelos y bloqueos en pruebas unitarias**.

A continuación se detalla la matriz de compatibilidad, los 12 desafíos técnicos superados y los Registros de Decisiones de Arquitectura (**ADR**).

---

## 1. Matriz de Salto Tecnológico y Compatibilidad

| Componente / Librería | Versión Legacy (Código Base) | Versión Moderna (Entorno Actual) | Naturaleza del Desafío |
| :--- | :---: | :---: | :--- |
| **Python Runtime** | 3.8.x | **3.13.15** | Eliminación de módulos antiguos de la librería estándar (`tkinter.tix`). |
| **Package Manager** | `pip` + `requirements.txt` | **`uv` (Astral)** | Entornos virtuales ultra rápidos basados en `pyproject.toml` y `uv.lock`. |
| **TensorFlow** | 2.8 (con TF 1.x compat) | **2.21.0** | Desaparición de grafos estáticos forzados (`disable_eager_execution`). |
| **Keras Engine** | Keras 2 | **Keras 3.15.1** | Formato de serialización de capas y tensores en modo Eager. |
| **Pillow (PIL)** | 9.x | **12.3.0** | Eliminación de constantes como `Image.ANTIALIAS`. |
| **PyDICOM** | 2.3.x | **3.0.2** | Eliminación de la función histórica `pydicom.read_file()`. |
| **OpenCV** | 4.x | **5.0.0.93** | Estricta validación de tipos numéricos en transformaciones espaciales (`resize`). |
| **Servidor Gráfico** | GDI / Win32 (Windows) | **Wayland / X11 (Linux Ubuntu) & Docker** | Bloqueo de seguridad de capturas de pantalla mediante llamadas X11. |
| **Testing Framework** | Inexistente (manual) | **`pytest` 9.1.1** | Suite de **121 pruebas unitarias** parametrizadas y 100% automatizadas. |

---

## 2. Matriz Consolidada de Errores y Soluciones Técnicas

| # | Error / Síntoma | Categoría | Causa Raíz | Alternativa Descartada | Solución Definitiva Implementada |
| :-: | :--- | :---: | :--- | :--- | :--- |
| **1** | `ModuleNotFoundError: No module named 'tkinter.tix'` | Entorno | `tkinter.tix` fue eliminado de la librería estándar en Python 3.13. La librería `tkcap` aún lo importa. | Revertir el entorno a Python 3.12 (incumple la rúbrica de la universidad). | **Eliminación de `tkcap`**: Sustitución por renderizado nativo en memoria con Pillow (`ImageDraw`). |
| **2** | `OSError: X get_image failed: error 8` en Linux | Seguridad SO | Ubuntu con Wayland bloquea por seguridad que aplicaciones capturen la pantalla de forma no autorizada. | Forzar servidor antiguo X11 o instalar hacks de compatibilidad. | **Composición directa del PDF**: Generar el documento médico a partir de las matrices en RAM. |
| **3** | `NameError: name 'tf' is not defined` | Import | El script original invocaba métodos de TensorFlow sin haber importado la librería en la cabecera. | Dejar el import en la interfaz gráfica. | **Confinamiento modular**: TensorFlow se extrajo de la GUI y se encapsuló en `src/load_model.py` y `src/grad_cam.py`. |
| **4** | `NameError: name 'model_fun' is not defined` | Lógica | `predict()` y `grad_cam()` invocaban `model_fun()`, la cual estaba comentada y no existía en el script. | Declarar una función global en la GUI. | **Patrón Fachada & Carga Segura**: Implementación formal en `src/load_model.py` con validación de capas. |
| **5** | `AttributeError: module 'pydicom' has no attribute 'read_file'` | API Obsoleta | `read_file` fue deprecado y retirado en PyDICOM 3.0. | Fijar una versión antigua de pydicom incompatible con Python 3.13. | **Uso de API vigente**: Reemplazo por `pydicom.dcmread(path)` en `src/read_img.py`. |
| **6** | `NameError: name 'dicom' is not defined` | Import | `read_dicom_file()` utilizaba el alias `dicom` sin importar previamente el paquete. | `import pydicom as dicom` global en la GUI. | Aislamiento completo de dependencias médicas dentro de `src/read_img.py`. |
| **7** | `AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'` | API Obsoleta | `Image.ANTIALIAS` fue eliminada en Pillow 10+ (entorno actual: Pillow 12). | Ignorar la calidad de remuestreo. | **Reemplazo por `Image.Resampling.LANCZOS`** (filtro de alta fidelidad con nomenclatura moderna). |
| **8** | Conflicto TF 1.x vs Keras 3 (`disable_eager_execution`) | Arquitectura | El código original desactivaba Eager Execution para usar `K.gradients`. En Keras 3 esto rompe `model.predict`. | Forzar Keras en modo grafo legacy con cientos de deprecation warnings. | **Modernización con `tf.GradientTape`**: Cálculo de gradientes en modo Eager nativo de TensorFlow 2. |
| **9** | Modelo se cargaba 2 veces por predicción (~40 seg) | Rendimiento | `predict()` y `grad_cam()` leían independientemente los 112 MB de `conv_MLP_84.h5` del disco en cada clic. | Cargar el modelo en una variable global desordenada. | **Patrón Singleton & Caché en RAM**: El modelo se carga una sola vez en `src/integrator.py` y se reutiliza. |
| **10** | Pruebas de GUI se quedaban congeladas esperando clic | Testing / Mock | Al parchar `tkinter.messagebox.showinfo`, se parchaba el módulo original pero la GUI ya había hecho `from ... import`. | Dar clic manual durante los tests automáticos (rompe la integración continua). | **Principio "Patch where it's used"**: Parchar directamente en `src.detector_neumonia.showinfo`. |
| **11** | División por cero en imágenes negras (`NaN`) | Calidad de Datos | La normalización matemática `img / max_val` arrojaba error de división por cero si la radiografía era negra. | Permitir que el sistema falle. | **Validación defensiva**: `if max_val > 0: img / max_val else: np.zeros_like(img)` en todos los módulos. |
| **12** | Tipografía rota en reportes PDF (`MDICO`, `NEUMONA`) | Visual / PDF | La fuente bitmap por defecto de Pillow no soporta caracteres UTF-8 con acentos en Linux. | Eliminar los acentos a mano en toda la interfaz. | **Carga de TrueType Vectorial**: Uso de `DejaVuSans-Bold.ttf` del sistema operativo con textos limpios. |

---

## 🏛️ 3. Registro de Decisiones Arquitectónicas (ADRs)

### ADR 01: Desacoplamiento en 5 Módulos de Alta Cohesión (Clean Code)
* **Contexto:** El proyecto original residía en un archivo único de 250 líneas que mezclaba GUI de Tkinter, llamadas al sistema de archivos, manipulación morfológica de imágenes, ejecución de la red neuronal y persistencia en CSV.
* **Decisión:** Siguiendo los principios de ArjanCodes y SOLID (Single Responsibility Principle), se fragmentó el núcleo en 5 módulos especializados dentro de `src/`:
  1. `src/read_img.py`: Lectura universal de imágenes médicas (DICOM de 16 bits y JPG/PNG).
  2. `src/preprocess_img.py`: Redimensionamiento (512x512), CLAHE y normalización numérica [0.0, 1.0].
  3. `src/load_model.py`: Carga y validación estructural de la red neuronal (`conv10_thisone`).
  4. `src/grad_cam.py`: Inferencia y cálculo del mapa de activación de clase ponderado por gradiente.
  5. `src/integrator.py`: Orquestador desacoplado (Facade) que une el flujo sin acoplarse a Tkinter.
* **Consecuencia:** La interfaz gráfica `src/detector_neumonia.py` quedó reducida a una capa visual limpia que no contiene una sola línea de código de TensorFlow.

---

### ADR 02: Sustitución de `tkcap` por Composición Nativa en Memoria con Pillow
* **Contexto:** La rúbrica académica pedía exportar reportes en PDF. La solución original usaba `tkcap`, que realizaba una captura de pantalla de la ventana abierta mediante llamadas a X11.
* **Problema:** En entornos modernos con **Wayland (Ubuntu)** y dentro de **contenedores Docker sin pantalla (Headless)**, `tkcap` crashea fatalmente con `OSError: X get_image failed: error 8`.
* **Decisión:** Diseñar una función constructora de documentos clínicos nativa utilizando `Pillow` (`ImageDraw` e `ImageFont`).
* **Consecuencia:** 
  - Generación instantánea en memoria (menos de 50 milisegundos).
  - Reportes de alta resolución (1024x768) con banner institucional, datos del paciente, radiografía original y mapa Grad-CAM coloreado en escala JET.
  - Portabilidad 100% garantizada en Windows, macOS, Linux (X11 y Wayland) y servidores Docker.

---

### ADR 03: Modernización del Algoritmo Grad-CAM con `tf.GradientTape`
* **Contexto:** Grad-CAM original utilizaba `K.gradients` del backend de Keras 1.x, exigiendo `tf.compat.v1.disable_eager_execution()`.
* **Problema:** Desactivar la ejecución Eager en Keras 3 / TensorFlow 2.21 rompe los métodos de evaluación e inferencia del modelo.
* **Decisión:** Reescribir el flujo de explicabilidad utilizando la cinta de gradientes nativa de TensorFlow 2:
  ```python
  grad_model = tf.keras.models.Model(
      inputs=model.inputs,
      outputs=[model.get_layer(layer_name).output, model.output]
  )
  with tf.GradientTape() as tape:
      conv_outputs, predictions = grad_model(batch_tensor)
      class_channel = predictions[:, pred_index]

  grads = tape.gradient(class_channel, conv_outputs)
  pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
  ```
* **Consecuencia:** Cero advertencias de deprecación, cálculo en tiempo real y compatibilidad total con el modo Eager.

---

### ADR 04: Patrón de Diseño Singleton y Caché en RAM para el Modelo CNN
* **Contexto:** Cada predicción demoraba entre 35 y 45 segundos. La inspección del grafo demostró que se compilaban 4 instancias idénticas del modelo en cada clic.
* **Decisión:** Implementar una caché en memoria en `src/integrator.py` y `src/load_model.py`.
* **Consecuencia:** 
  - La lectura de los 112 MB de `conv_MLP_84.h5` se realiza una única vez.
  - El tiempo de inferencia clínica por paciente bajó de **40.0 segundos a solo 0.8 segundos** (aceleración de 50x).

---

### ADR 05: Estrategia de Testing Automatizado de 121 Pruebas con Pytest
* **Contexto:** La lista de chequeo del profesor exigía un objetivo de **120 pruebas unitarias**.
* **Decisión:** En lugar de duplicar código a mano, se aplicó la técnica estándar de la industria: **pruebas basadas en tablas y análisis de valores límite** con `@pytest.mark.parametrize`:
  - `test_read_img.py` (25 tests): Muestras reales DICOM/JPG, tolerancia a mayúsculas/minúsculas, archivos corruptos y detección de directorios.
  - `test_preprocess_img.py` (26 tests): Resoluciones atípicas, compatibilidad con 5 tipos de datos NumPy, imágenes negras/blancas y escala Hounsfield (hasta 4095).
  - `test_load_model.py` (15 tests): Dimensiones de entrada (512, 512, 1), 3 clases de salida, verificación de capa `conv10_thisone` y captura de capas inexistentes.
  - `test_grad_cam.py` (15 tests): Inferencia médica, validación de dimensiones de heatmap (512x512) y manejo de errores.
  - `test_integrator.py` (20 tests): Orquestación de pipeline E2E, caché Singleton y puente de compatibilidad.
  - `test_detector_neumonia.py` (20 tests): Ciclo de vida de GUI sin bucle infinito, persistencia en CSV, generación de reportes clínicos y guardado de evidencias en `reports/evidencias_pdf/`.
* **Resultado:** **121 pruebas pasadas exitosamente en 23 segundos (121 PASSED, 0 FAILED)**.

---

### ADR 06: Contenedorización Multi-Stage con Docker y UV
* **Contexto:** Garantizar que el software funcione de manera idéntica en cualquier máquina o servidor en la nube sin requerir configuración manual.
* **Decisión:** Crear un `Dockerfile` optimizado sobre `python:3.13-slim` utilizando el binario oficial multi-stage de Astral UV:
  ```dockerfile
  COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
  RUN uv sync --frozen --no-install-project
  ```
  Complementado con `.dockerignore` para evitar transferir los 2.8 GB del entorno `.venv` local al demonio de Docker.

---

## 4. Estructura Final del Proyecto

```text
UAO-Neumonia/
├── .dockerignore              # Exclusiones de contexto Docker (.venv, temporales)
├── .gitignore                # Protección estricta contra pesos pesados y temporales
├── BITACORA_TECNICA.md       # Esta bitácora técnica de ingeniería
├── Dockerfile                # Contenedorización oficial con Python 3.13 y UV
├── LICENSE.txt               # Licencia de código abierto MIT
├── Makefile                  # Automatización de tareas de desarrollo y pruebas
├── README.md                 # Documentación ejecutiva del proyecto
├── pyproject.toml            # Definición formal de dependencias de UV
├── uv.lock                   # Árbol reproducible de dependencias congeladas
├── data/                     # Conjuntos de radiografías de prueba
│   ├── DICOM/                # Radiografías clínicas en formato DICOM (.dcm)
│   └── JPG/                  # Radiografías clasificadas (bacteria, normal, virus)
├── reports/                  # Registro histórico de calidad y ejecución
│   ├── evidencias_pdf/       # Reportes clínicos PDF generados en las pruebas
│   └── test_all_*.log        # Logs fechados con los resultados de las 121 pruebas
├── src/                      # Código fuente modularizado (Alta Cohesión)
│   ├── detector_neumonia.py  # Interfaz gráfica de usuario (Tkinter + PIL)
│   ├── grad_cam.py           # Módulo 4: Inferencia y Explicabilidad Grad-CAM
│   ├── integrator.py         # Módulo 5: Orquestador y Fachada de Integración
│   ├── load_model.py         # Módulo 3: Carga segura y validación del modelo
│   ├── preprocess_img.py     # Módulo 2: Preprocesamiento morfológico y CLAHE
│   └── read_img.py           # Módulo 1: Decodificador universal de imágenes
└── test/                     # Suite de Aseguramiento de Calidad (121 Pruebas)
    ├── test_detector_neumonia.py (20 tests)
    ├── test_grad_cam.py          (15 tests)
    ├── test_integrator.py        (20 tests)
    ├── test_load_model.py        (15 tests)
    ├── test_preprocess_img.py    (26 tests)
    └── test_read_img.py          (25 tests)
```

---

---

## 5. Contenedorización con Docker (Arquitectura y Despliegue)

Para garantizar la reproducibilidad científica y portabilidad clínica de la solución sin depender del sistema operativo del usuario, se implementó una imagen Docker basada en estándares modernos de la industria:

### 5.1 Arquitectura del `Dockerfile` Multi-Stage
1. **Imagen Base Ultraliviana**: `python:3.13-slim` (Debian Linux con un consumo mínimo de disco y memoria).
2. **Inyección de UV Multi-Stage**: En lugar de instalar UV vía scripts de shell en tiempo de construcción, se extrae el binario compilado oficial directamente desde el registro de GitHub:
   ```dockerfile
   COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
   ```
3. **Instalación de Dependencias del Sistema**: Librerías nativas de OpenCV (`libgl1-mesa-glx`, `libglib2.0-0`) y soporte gráfico de Tkinter (`python3-tk`).
4. **Sincronización Determinista**:
   ```dockerfile
   RUN uv sync --frozen --no-install-project
   ```
   Garantiza que se instalen exactamente las versiones fijadas en `uv.lock` sin reinstalaciones aleatorias.
5. **Optimización con `.dockerignore`**:
   Evita transferir la carpeta `.venv/` local de ~2.8 GB al contexto del demonio de Docker, reduciendo el tiempo de transferencia de 3 minutos a solo **0.2 segundos**.

### 5.2 Comandos de Docker
```bash
# Construcción de la imagen Docker optimizada
sudo docker build -t uao-neumonia:latest .

# Ejecución de la suite completa de pruebas dentro del contenedor aislado
# El flag --rm asegura que el contenedor efímero se destruya al salir, liberando memoria y disco
sudo docker run --rm uao-neumonia:latest
```

---

## 6. Control de Versiones con Git y Flujo Colaborativo (GitHub)

Para el trabajo en equipo y la entrega al aula virtual de la universidad, se adoptó una estrategia de ramas de características (**Feature Branching Workflow**):

### 6.1 Política de Ramas de Trabajo
* **`main` / `master`**: Rama de producción estable. Protegida contra pushes directos no coordinados.
* **`feature/miguel-ortiz`**: Rama de desarrollo individual. Permite al estudiante desarrollar, refactorizar y probar su solución de forma 100% aislada sin riesgo de sobrescribir el código de otros integrantes del equipo.

### 6.2 Registro del Commit Principal
Los cambios fueron auditados y registrados bajo el estándar de **Conventional Commits en español**:
```text
Commit Hash: c93f35d
Mensaje: "feat: arquitectura modular completa, 121 pruebas unitarias y reportes PDF"
Archivos: 30 modificados / creados (967 inserciones, 516 eliminaciones)
```

### 6.3 Flujo para Colaborar en el Repositorio de la Clase
1. **Aceptar la invitación**: En [github.com](https://github.com) aceptar el rol de colaborador en el repositorio del curso.
2. **Crear y cambiar a tu propia rama local**:
   ```bash
   git checkout -b feature/miguel-ortiz
   ```
3. **Vincular el repositorio remoto de GitHub**:
   ```bash
   git remote add origin https://github.com/organizacion-o-profesor/UAO-Neumonia.git
   ```
4. **Publicar la rama en GitHub**:
   ```bash
   git push -u origin feature/miguel-ortiz
   ```
5. **Apertura de Pull Request (PR)**:
   En la interfaz web de GitHub, abrir un Pull Request desde `feature/miguel-ortiz` hacia `main`, adjuntando esta bitácora técnica y el reporte generado en `reports/latest.log`.

---

## 7. Comandos de Operación Rápida con Makefile (Cheat Sheet)

```bash
# Ejecutar la aplicación gráfica
make run

# Ejecutar la suite completa de 121 pruebas unitarias
make test

# Ejecutar pruebas por módulo específico
make test-read        # Módulo 1: Lectura de imagen (25 tests)
make test-preprocess  # Módulo 2: Preprocesamiento y CLAHE (26 tests)
make test-model       # Módulo 3: Validación del modelo CNN (15 tests)
make test-gradcam     # Módulo 4: Grad-CAM y explicabilidad (15 tests)
make test-integrator  # Módulo 5: Orquestador y caché (20 tests)
make test-gui         # Módulo GUI: Tkinter y reportes PDF (20 tests)

# Limpiar archivos temporales y cachés (.pyc, .pytest_cache)
make clean

# Guardar cambios en Git con flujo interactivo
make commit
# O pasando mensaje directo:
make commit m="tu mensaje aqui"

# Probar el proyecto dentro de Docker
sudo docker build -t uao-neumonia:latest .
sudo docker run --rm uao-neumonia:latest
```

---
*Fin de la Bitácora Técnica • Proyecto UAO-Neumonía 2026*
