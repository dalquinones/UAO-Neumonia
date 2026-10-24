warning: in the working copy of 'README.md', LF will be replaced by CRLF the next time Git touches it
[1mdiff --git a/Dockerfile b/Dockerfile[m
[1mindex 44fd97d..b2143b5 100644[m
[1m--- a/Dockerfile[m
[1m+++ b/Dockerfile[m
[36m@@ -1,9 +1,21 @@[m
[31m-FROM python:latest[m
[32m+[m[32mFROM python:3.13-slim[m
 [m
[31m-RUN apt-get update -y && \[m
[31m-    apt-get install python3-opencv -y [m
[32m+[m[32m# Dependencias del sistema: OpenCV y Tkinter necesitan estas librerias[m
[32m+[m[32mRUN apt-get update && apt-get install -y --no-install-recommends \[m
[32m+[m[32m    libgl1 \[m
[32m+[m[32m    libglib2.0-0 \[m
[32m+[m[32m    tk \[m
[32m+[m[32m    && rm -rf /var/lib/apt/lists/*[m
 [m
[31m-WORKDIR /home/src[m
[32m+[m[32mWORKDIR /app[m
 [m
[31m-COPY . ./[m
[31m-RUN pip install -r requirements.txt[m
[32m+[m[32m# Se copian primero las dependencias para aprovechar la cache de capas[m
[32m+[m[32mCOPY requirements.txt .[m
[32m+[m[32mRUN pip install --no-cache-dir -r requirements.txt[m
[32m+[m
[32m+[m[32mCOPY . .[m
[32m+[m
[32m+[m[32mENV TF_USE_LEGACY_KERAS=1[m
[32m+[m[32mENV TF_CPP_MIN_LOG_LEVEL=2[m
[32m+[m
[32m+[m[32mENTRYPOINT ["python", "detector_neumonia.py"][m
[1mdiff --git a/README.md b/README.md[m
[1mindex 0f6187c..a0ab5e0 100644[m
[1m--- a/README.md[m
[1m+++ b/README.md[m
[36m@@ -1,105 +1,354 @@[m
[31m-## Hola! Bienvenido a la herramienta para la detección rápida de neumonía[m
[32m+[m[32m# Detección de Neumonía en Radiografías de Tórax[m
 [m
[31m-Deep Learning aplicado en el procesamiento de imágenes radiográficas de tórax en formato DICOM con el fin de clasificarlas en 3 categorías diferentes:[m
[32m+[m[32m![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)[m
[32m+[m[32m![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?logo=tensorflow&logoColor=white)[m
[32m+[m[32m![uv](https://img.shields.io/badge/gestor-uv-DE5FE9)[m
[32m+[m[32m![Tests](https://img.shields.io/badge/pytest-13%20passed-12695E)[m
[32m+[m[32m![License](https://img.shields.io/badge/licencia-MIT-blue)[m
 [m
[31m-1. Neumonía Bacteriana[m
[32m+[m[32mHerramienta de apoyo al diagnóstico médico que clasifica radiografías de tórax en[m
[32m+[m[32m**neumonía bacteriana**, **neumonía viral** o **sin neumonía**, y explica su decisión[m
[32m+[m[32mmediante un mapa de calor Grad-CAM superpuesto sobre la imagen original.[m
 [m
[31m-2. Neumonía Viral[m
[32m+[m[32m> **Aviso**: este software es un ejercicio académico. No constituye un dispositivo[m
[32m+[m[32m> médico ni debe usarse para tomar decisiones clínicas.[m
 [m
[31m-3. Sin Neumonía[m
[32m+[m[32m---[m
[32m+[m
[32m+[m[32m## Tabla de contenido[m
[32m+[m
[32m+[m[32m- [Características](#características)[m
[32m+[m[32m- [Árbol del proyecto](#árbol-del-proyecto)[m
[32m+[m[32m- [Arquitectura](#arquitectura)[m
[32m+[m[32m- [Flujo de datos](#flujo-de-datos)[m
[32m+[m[32m- [Requisitos](#requisitos)[m
[32m+[m[32m- [Instalación](#instalación)[m
[32m+[m[32m- [Ejecución](#ejecución)[m
[32m+[m[32m- [Ejecución con Docker](#ejecución-con-docker)[m
[32m+[m[32m- [Uso de la aplicación](#uso-de-la-aplicación)[m
[32m+[m[32m- [Pruebas](#pruebas)[m
[32m+[m[32m- [Módulos](#módulos)[m
[32m+[m[32m- [Decisiones de diseño](#decisiones-de-diseño)[m
[32m+[m[32m- [Errores corregidos del código base](#errores-corregidos-del-código-base)[m
[32m+[m[32m- [Limitaciones conocidas](#limitaciones-conocidas)[m
[32m+[m[32m- [Licencia](#licencia)[m
[32m+[m[32m- [Autores](#autores)[m
[32m+[m
[32m+[m[32m---[m
[32m+[m
[32m+[m[32m## Características[m
[32m+[m
[32m+[m[32m- Lectura de imágenes en formato **DICOM** y **JPG/PNG**.[m
[32m+[m[32m- Preprocesamiento estándar: redimensionamiento a 512×512, escala de grises,[m
[32m+[m[32m  ecualización adaptativa **CLAHE** y normalización.[m
[32m+[m[32m- Inferencia con una red convolucional entrenada (`conv_MLP_84.h5`, arquitectura[m
[32m+[m[32m  `Net5Blocks`, 57 capas).[m
[32m+[m[32m- Explicabilidad con **Grad-CAM** implementado sobre `tf.GradientTape`.[m
[32m+[m[32m- Interfaz gráfica en Tkinter con exportación de resultados a **CSV** y **PDF**.[m
[32m+[m[32m- Suite de pruebas unitarias con **pytest** que se ejecuta sin necesidad del modelo.[m
 [m
[31m-Aplicación de una técnica de explicación llamada Grad-CAM para resaltar con un mapa de calor las regiones relevantes de la imagen de entrada.[m
[32m+[m[32m---[m
[32m+[m
[32m+[m[32m## Árbol del proyecto[m
[32m+[m
[32m+[m[32m```text[m
[32m+[m[32mUAO-Neumonia/[m
[32m+[m[32m├── src/                       Lógica del pipeline[m
[32m+[m[32m│   ├── __init__.py[m
[32m+[m[32m│   ├── read_img.py            Lectura de DICOM y JPG a arreglos NumPy[m
[32m+[m[32m│   ├── preprocess_img.py      Redimensionado, CLAHE y normalización[m
[32m+[m[32m│   ├── load_model.py          Carga del modelo con caché[m
[32m+[m[32m│   ├── grad_cam.py            Mapa de calor con GradientTape[m
[32m+[m[32m│   └── integrator.py          Orquestación del pipeline completo[m
[32m+[m[32m├── tests/                     Pruebas unitarias[m
[32m+[m[32m│   ├── __init__.py[m
[32m+[m[32m│   ├── conftest.py            Fixtures: imágenes y DICOM sintéticos[m
[32m+[m[32m│   ├── test_read_img.py[m
[32m+[m[32m│   ├── test_preprocess_img.py[m
[32m+[m[32m│   ├── test_load_model.py[m
[32m+[m[32m│   └── test_integrator.py[m
[32m+[m[32m├── detector_neumonia.py       Interfaz gráfica (Tkinter)[m
[32m+[m[32m├── compat_tix.py              Compatibilidad de tkcap con Python 3.13[m
[32m+[m[32m├── conftest.py                Habilita la raíz en el path de pytest[m
[32m+[m[32m├── Dockerfile[m
[32m+[m[32m├── Makefile                   Automatización de tareas[m
[32m+[m[32m├── pyproject.toml             Configuración del proyecto y de pytest[m
[32m+[m[32m├── requirements.txt           Dependencias con versiones fijadas[m
[32m+[m[32m├── uv.lock                    Resolución exacta del entorno[m
[32m+[m[32m├── LICENSE.md[m
[32m+[m[32m└── README.md[m
[32m+[m[32m```[m
[32m+[m
[32m+[m[32mLos archivos de pesos (`conv_MLP_84.h5`, `WilhemNet86.h5`) y las salidas de la[m
[32m+[m[32maplicación (`historial.csv`, `Reporte*.pdf`) están excluidos del repositorio.[m
[32m+[m
[32m+[m[32m---[m
[32m+[m
[32m+[m[32m## Arquitectura[m
[32m+[m
[32m+[m[32mCada módulo tiene una responsabilidad única y no conoce los detalles internos de[m
[32m+[m[32mlos demás. La interfaz gráfica solo depende de `integrator` y de `read_img`.[m
[32m+[m
[32m+[m[32m```mermaid[m
[32m+[m[32mflowchart TD[m
[32m+[m[32m    GUI["detector_neumonia.py<br/>Interfaz gráfica"][m
[32m+[m[32m    INT["integrator.py<br/>Orquestación"][m
[32m+[m[32m    READ["read_img.py<br/>Lectura"][m
[32m+[m[32m    PRE["preprocess_img.py<br/>Preprocesamiento"][m
[32m+[m[32m    LOAD["load_model.py<br/>Carga del modelo"][m
[32m+[m[32m    CAM["grad_cam.py<br/>Explicabilidad"][m
[32m+[m
[32m+[m[32m    GUI -->|ruta del archivo| READ[m
[32m+[m[32m    GUI -->|arreglo RGB| INT[m
[32m+[m[32m    INT --> PRE[m
[32m+[m[32m    INT --> LOAD[m
[32m+[m[32m    INT -->|modelo ya cargado| CAM[m
[32m+[m[32m    CAM --> PRE[m
[32m+[m[32m    CAM --> LOAD[m
[32m+[m[32m```[m
[32m+[m
[32m+[m[32mEl modelo se carga **una sola vez** por ejecución. `integrator` lo obtiene de[m
[32m+[m[32m`load_model` y lo inyecta en `grad_cam`, en lugar de que cada módulo lo lea del[m
[32m+[m[32mdisco por su cuenta.[m
 [m
 ---[m
 [m
[31m-## Uso de la herramienta:[m
[32m+[m[32m## Flujo de datos[m
[32m+[m
[32m+[m[32m```mermaid[m
[32m+[m[32msequenceDiagram[m
[32m+[m[32m    participant U as Usuario[m
[32m+[m[32m    participant G as Interfaz[m
[32m+[m[32m    participant I as integrator[m
[32m+[m[32m    participant M as Modelo[m
[32m+[m
[32m+[m[32m    U->>G: Selecciona archivo DICOM[m
[32m+[m[32m    G->>G: read_dicom_file() → arreglo RGB[m
[32m+[m[32m    G->>U: Muestra la radiografía[m
[32m+[m[32m    U->>G: Clic en "Predecir"[m
[32m+[m[32m    G->>I: predict(arreglo)[m
[32m+[m[32m    I->>I: preprocess() → lote (1, 512, 512, 1)[m
[32m+[m[32m    I->>M: Inferencia[m
[32m+[m[32m    M-->>I: Vector de 3 probabilidades[m
[32m+[m[32m    I->>I: grad_cam() → mapa de calor[m
[32m+[m[32m    I-->>G: (etiqueta, probabilidad, heatmap)[m
[32m+[m[32m    G->>U: Resultado, probabilidad y mapa de calor[m
[32m+[m[32m```[m
 [m
[31m-A continuación le explicaremos cómo empezar a utilizarla.[m
[32m+[m[32m---[m
 [m
[31m-Requerimientos necesarios para el funcionamiento:[m
[32m+[m[32m## Requisitos[m
 [m
[31m-- Instale Anaconda para Windows siguiendo las siguientes instrucciones:[m
[31m-  https://docs.anaconda.com/anaconda/install/windows/[m
[32m+[m[32m| Componente | Versión |[m
[32m+[m[32m|---|---|[m
[32m+[m[32m| Python | 3.13 |[m
[32m+[m[32m| uv | 0.12 o superior |[m
[32m+[m[32m| Modelo entrenado | `conv_MLP_84.h5` en la raíz del proyecto |[m
[32m+[m[32m| Docker | 20.10 o superior (opcional) |[m
[32m+[m[32m| GNU Make | 4.4 o superior (opcional) |[m
 [m
[31m-- Abra Anaconda Prompt y ejecute las siguientes instrucciones:[m
[32m+[m[32mEn Windows, `make` no viene incluido. Se instala con[m
[32m+[m[32m[Chocolatey](https://chocolatey.org/install): `choco install make`.[m
 [m
[31m-  conda create -n tf tensorflow[m
[32m+[m[32m---[m
 [m
[31m-  conda activate tf[m
[32m+[m[32m## Instalación[m
 [m
[31m-  cd UAO-Neumonia[m
[32m+[m[32m```bash[m
[32m+[m[32mgit clone https://github.com/marlonvalenciamentor-eng/UAO-Neumonia.git[m
[32m+[m[32mcd UAO-Neumonia[m
 [m
[31m-  pip install -r requirements.txt[m
[32m+[m[32muv venv --python 3.13[m
[32m+[m[32muv pip install -r requirements.txt[m
[32m+[m[32m```[m
 [m
[31m-  python detector_neumonia.py[m
[32m+[m[32mO con Make:[m
 [m
[31m-Uso de la Interfaz Gráfica:[m
[32m+[m[32m```bash[m
[32m+[m[32mmake install[m
[32m+[m[32m```[m
 [m
[31m-- Ingrese la cédula del paciente en la caja de texto[m
[31m-- Presione el botón 'Cargar Imagen', seleccione la imagen del explorador de archivos del computador (Imagenes de prueba en https://drive.google.com/drive/folders/1WOuL0wdVC6aojy8IfssHcqZ4Up14dy0g?usp=drive_link)[m
[31m-- Presione el botón 'Predecir' y espere unos segundos hasta que observe los resultados[m
[31m-- Presione el botón 'Guardar' para almacenar la información del paciente en un archivo excel con extensión .csv[m
[31m-- Presione el botón 'PDF' para descargar un archivo PDF con la información desplegada en la interfaz[m
[31m-- Presión el botón 'Borrar' si desea cargar una nueva imagen[m
[32m+[m[32mColoque el archivo `conv_MLP_84.h5` en la raíz del proyecto. No está incluido en[m
[32m+[m[32mel repositorio por su tamaño (112 MB).[m
 [m
 ---[m
 [m
[31m-## Arquitectura de archivos propuesta.[m
[32m+[m[32m## Ejecución[m
 [m
[31m-## detector_neumonia.py[m
[32m+[m[32m```bash[m
[32m+[m[32muv run detector_neumonia.py[m
[32m+[m[32m```[m
 [m
[31m-Contiene el diseño de la interfaz gráfica utilizando Tkinter.[m
[32m+[m[32mO con Make:[m
 [m
[31m-Los botones llaman métodos contenidos en otros scripts.[m
[32m+[m[32m```bash[m
[32m+[m[32mmake run[m
[32m+[m[32m```[m
 [m
[31m-## integrator.py[m
[32m+[m[32m---[m
 [m
[31m-Es un módulo que integra los demás scripts y retorna solamente lo necesario para ser visualizado en la interfaz gráfica.[m
[31m-Retorna la clase, la probabilidad y una imagen el mapa de calor generado por Grad-CAM.[m
[32m+[m[32m## Ejecución con Docker[m
 [m
[31m-## read_img.py[m
[32m+[m[32m```bash[m
[32m+[m[32m# Construir la imagen[m
[32m+[m[32mdocker build -t neumonia .[m
 [m
[31m-Script que lee la imagen en formato DICOM para visualizarla en la interfaz gráfica. Además, la convierte a arreglo para su preprocesamiento.[m
[32m+[m[32m# Ejecutar montando la carpeta de datos[m
[32m+[m[32mdocker run -v $(pwd)/data:/app/data neumonia[m
[32m+[m[32m```[m
 [m
[31m-## preprocess_img.py[m
[32m+[m[32mO con Make:[m
 [m
[31m-Script que recibe el arreglo proveniento de read_img.py, realiza las siguientes modificaciones:[m
[32m+[m[32m```bash[m
[32m+[m[32mmake docker-build[m
[32m+[m[32mmake docker-run[m
[32m+[m[32m```[m
 [m
[31m-- resize a 512x512[m
[31m-- conversión a escala de grises[m
[31m-- ecualización del histograma con CLAHE[m
[31m-- normalización de la imagen entre 0 y 1[m
[31m-- conversión del arreglo de imagen a formato de batch (tensor)[m
[32m+[m[32mLa interfaz gráfica requiere un servidor X11. En Linux se expone el display del[m
[32m+[m[32manfitrión; en Windows y macOS se recomienda usar el contenedor únicamente para[m
[32m+[m[32mprocesamiento por lotes.[m
 [m
[31m-## load_model.py[m
[32m+[m[32m---[m
 [m
[31m-Script que lee el archivo binario del modelo de red neuronal convolucional previamente entrenado llamado 'WilhemNet86.h5'.[m
[32m+[m[32m## Uso de la aplicación[m
 [m
[31m-## grad_cam.py[m
[32m+[m[32m1. **Cargar Imagen** — abre el selector de archivos y muestra la radiografía.[m
[32m+[m[32m2. **Predecir** — ejecuta el pipeline y muestra la clase, la probabilidad y el[m
[32m+[m[32m   mapa de calor.[m
[32m+[m[32m3. **Guardar** — agrega el resultado a `historial.csv`.[m
[32m+[m[32m4. **PDF** — genera un reporte con captura de la ventana.[m
[32m+[m[32m5. **Borrar** — limpia la interfaz para un nuevo caso.[m
 [m
[31m-Script que recibe la imagen y la procesa, carga el modelo, obtiene la predicción y la capa convolucional de interés para obtener las características relevantes de la imagen.[m
[32m+[m[32m<!-- Reemplace por su propia captura: guárdela en docs/captura.png -->[m
[32m+[m[32m<!-- ![Interfaz de la aplicación](docs/captura.png) -->[m
 [m
 ---[m
 [m
[31m-## Acerca del Modelo[m
[32m+[m[32m## Pruebas[m
[32m+[m
[32m+[m[32m```bash[m
[32m+[m[32muv run pytest -v[m
[32m+[m[32m```[m
[32m+[m
[32m+[m[32mO con Make:[m
[32m+[m
[32m+[m[32m```bash[m
[32m+[m[32mmake test[m
[32m+[m[32m```[m
 [m
[31m-La red neuronal convolucional implementada (CNN) es basada en el modelo implementado por F. Pasa, V.Golkov, F. Pfeifer, D. Cremers & D. Pfeifer[m
[31m-en su artículo Efcient Deep Network Architectures for Fast Chest X-Ray Tuberculosis Screening and Visualization.[m
[32m+[m[32mLa suite cubre las cuatro áreas del pipeline:[m
 [m
[31m-Está compuesta por 5 bloques convolucionales, cada uno contiene 3 convoluciones; dos secuenciales y una conexión 'skip' que evita el desvanecimiento del gradiente a medida que se avanza en profundidad.[m
[31m-Con 16, 32, 48, 64 y 80 filtros de 3x3 para cada bloque respectivamente.[m
[32m+[m[32m| Archivo | Qué verifica |[m
[32m+[m[32m|---|---|[m
[32m+[m[32m| `test_read_img.py` | Forma y rango de los arreglos leídos desde DICOM y JPG, y el error ante archivos inexistentes |[m
[32m+[m[32m| `test_preprocess_img.py` | Forma del lote, normalización a 0–1, independencia del tamaño de entrada y determinismo |[m
[32m+[m[32m| `test_load_model.py` | Error ante rutas inválidas, efectividad de la caché y presencia de la capa usada por Grad-CAM |[m
[32m+[m[32m| `test_integrator.py` | Cobertura de las tres clases y contrato de salida del pipeline completo |[m
 [m
[31m-Después de cada bloque convolucional se encuentra una capa de max pooling y después de la última una capa de Average Pooling seguida por tres capas fully-connected (Dense) de 1024, 1024 y 3 neuronas respectivamente.[m
[32m+[m[32mLas pruebas que necesitan el modelo entrenado se **omiten automáticamente** si el[m
[32m+[m[32marchivo `.h5` no está presente, de modo que la suite se ejecuta en cualquier[m
[32m+[m[32mmáquina y en entornos de integración continua.[m
[32m+[m
[32m+[m[32m---[m
[32m+[m
[32m+[m[32m## Módulos[m
[32m+[m
[32m+[m[32m| Módulo | Responsabilidad | Entrada | Salida |[m
[32m+[m[32m|---|---|---|---|[m
[32m+[m[32m| `read_img` | Lectura de archivos | Ruta a DICOM o JPG | Arreglo RGB e imagen PIL |[m
[32m+[m[32m| `preprocess_img` | Acondicionamiento | Arreglo RGB | Lote `(1, 512, 512, 1)` normalizado |[m
[32m+[m[32m| `load_model` | Acceso al modelo | Ruta al `.h5` | Modelo Keras en caché |[m
[32m+[m[32m| `grad_cam` | Explicabilidad | Arreglo RGB y modelo | Imagen con mapa de calor |[m
[32m+[m[32m| `integrator` | Orquestación | Arreglo RGB | Etiqueta, probabilidad y mapa |[m
[32m+[m
[32m+[m[32m---[m
 [m
[31m-Para regularizar el modelo utilizamos 3 capas de Dropout al 20%; dos en los bloques 4 y 5 conv y otra después de la 1ra capa Dense.[m
[32m+[m[32m## Decisiones de diseño[m
[32m+[m
[32m+[m[32m**Carga única del modelo.** El código base cargaba el archivo `.h5` dos veces por[m
[32m+[m[32mpredicción: una en `predict()` y otra dentro de `grad_cam()`. Con dos[m
[32m+[m[32mpredicciones consecutivas se observaron cuatro cargas del grafo. La caché de[m
[32m+[m[32m`load_model` y la inyección del modelo en `grad_cam` reducen esto a una sola[m
[32m+[m[32mlectura por ejecución.[m
[32m+[m
[32m+[m[32m**Grad-CAM con `GradientTape`.** La implementación original usaba `K.gradients` y[m
[32m+[m[32m`K.function`, que exigen el modo grafo de TensorFlow 1 mediante[m
[32m+[m[32m`disable_eager_execution()`. Esa dependencia generaba una docena de advertencias[m
[32m+[m[32mde deprecación en cada ejecución. La reescritura con `tf.GradientTape` es el API[m
[32m+[m[32mvigente, funciona en modo eager y eliminó esas advertencias.[m
[32m+[m
[32m+[m[32m**Constantes con nombre.** Valores como el tamaño de entrada, los parámetros de[m
[32m+[m[32mCLAHE y el nombre de la capa convolucional dejaron de estar incrustados en el[m
[32m+[m[32mcuerpo de las funciones. El nombre de la capa además es parámetro de `grad_cam`,[m
[32m+[m[32mde modo que cambiar de modelo no obliga a editar el módulo.[m
[32m+[m
[32m+[m[32m**Validación de bordes.** La normalización original dividía por `array.max()` sin[m
[32m+[m[32mverificar. Una imagen completamente negra producía una división por cero; ahora[m
[32m+[m[32mse devuelve un arreglo de ceros.[m
[32m+[m
[32m+[m[32m**Separación de la interfaz.** `detector_neumonia.py` quedó reducido a la clase[m
[32m+[m[32m`App`. No importa TensorFlow, OpenCV ni pydicom: solo conoce `integrator.predict`[m
[32m+[m[32my `read_img.read_dicom_file`.[m
[32m+[m
[32m+[m[32m---[m
[32m+[m
[32m+[m[32m## Errores corregidos del código base[m
[32m+[m
[32m+[m[32mEl proyecto original fue escrito para Python 3.8, TensorFlow 2.8, Pillow 9 y[m
[32m+[m[32mpydicom 2.3. Migrarlo a Python 3.13 requirió resolver los siguientes problemas:[m
[32m+[m
[32m+[m[32m| # | Síntoma | Causa | Solución |[m
[32m+[m[32m|---|---|---|---|[m
[32m+[m[32m| 1 | `ModuleNotFoundError: tkinter.tix` | El módulo fue eliminado en Python 3.13; `tkcap` aún lo importa | Módulo de compatibilidad `compat_tix.py` |[m
[32m+[m[32m| 2 | `NameError: name 'tf' is not defined` | Faltaba `import tensorflow as tf` | Import agregado |[m
[32m+[m[32m| 3 | `NameError: name 'model_fun' is not defined` | La función de carga del modelo no estaba definida | Implementada y luego trasladada a `load_model.py` |[m
[32m+[m[32m| 4 | `AttributeError: read_file` | `pydicom.read_file` fue eliminado en pydicom 3.0 | Reemplazado por `dcmread` |[m
[32m+[m[32m| 5 | `NameError: name 'dicom' is not defined` | Faltaba `import pydicom as dicom` | Import agregado |[m
[32m+[m[32m| 6 | `NameError: name 'K' is not defined` | Faltaba el backend de Keras usado por Grad-CAM | Resuelto de raíz al reescribir con `GradientTape` |[m
[32m+[m[32m| 7 | `AttributeError: Image.ANTIALIAS` | Eliminado en Pillow 10 | Reemplazado por `Image.Resampling.LANCZOS` |[m
[32m+[m[32m| 8 | Inferencia lenta | El modelo se cargaba dos veces por predicción | Caché e inyección de dependencia |[m
[32m+[m
[32m+[m[32mAdicionalmente, los pesos `.h5` fueron guardados con Keras 2 y no son legibles por[m
[32m+[m[32mKeras 3, que es la versión que acompaña a TensorFlow 2.21. Se resolvió instalando[m
[32m+[m[32m`tf-keras` y activando `TF_USE_LEGACY_KERAS=1` antes de importar TensorFlow.[m
[32m+[m
[32m+[m[32m---[m
[32m+[m
[32m+[m[32m## Limitaciones conocidas[m
[32m+[m
[32m+[m[32m**El mapa de calor puede activarse fuera del área pulmonar.** En imágenes que son[m
[32m+[m[32mfotografías de radiografías impresas —con marco, rotación y artefactos de borde—[m
[32m+[m[32mse observaron activaciones concentradas en las esquinas y el contorno de la placa,[m
[32m+[m[32mno en el parénquima pulmonar, pese a arrojar probabilidades superiores al 85 %.[m
[32m+[m[32mUna predicción con esas características **no es confiable**, y es precisamente lo[m
[32m+[m[32mque Grad-CAM permite detectar. Se recomienda verificar el mapa de calor antes de[m
[32m+[m[32mconsiderar válido cualquier resultado.[m
[32m+[m
[32m+[m[32m**Advertencias residuales de dependencias.** Persisten tres advertencias de[m
[32m+[m[32mdeprecación emitidas por `tf-keras` y `gast`, ambas dependencias internas de[m
[32m+[m[32mTensorFlow. No provienen del código del proyecto y se resolverían migrando los[m
[32m+[m[32mpesos al formato nativo `.keras` de Keras 3, lo que alteraría el modelo entregado[m
[32m+[m[32my queda fuera del alcance de este trabajo.[m
[32m+[m
[32m+[m[32m**`python-xlib` no tiene función en Windows.** Se conserva en `requirements.txt`[m
[32m+[m[32mporque `pyautogui` la necesita en Linux, que es el sistema base del contenedor[m
[32m+[m[32mDocker.[m
[32m+[m
[32m+[m[32m---[m
[32m+[m
[32m+[m[32m## Licencia[m
[32m+[m
[32m+[m[32mDistribuido bajo la licencia MIT. Consulte el archivo [LICENSE.md](LICENSE.md).[m
[32m+[m
[32m+[m[32m---[m
 [m
[31m-## Acerca de Grad-CAM[m
[32m+[m[32m## Autores[m
 [m
[31m-Es una técnica utilizada para resaltar las regiones de una imagen que son importantes para la clasificación. Un mapeo de activaciones de clase para una categoría en particular indica las regiones de imagen relevantes utilizadas por la CNN para identificar esa categoría.[m
[32m+[m[32m<!-- Complete con los integrantes del grupo -->[m
 [m
[31m-Grad-CAM realiza el cálculo del gradiente de la salida correspondiente a la clase a visualizar con respecto a las neuronas de una cierta capa de la CNN. Esto permite tener información de la importancia de cada neurona en el proceso de decisión de esa clase en particular. Una vez obtenidos estos pesos, se realiza una combinación lineal entre el mapa de activaciones de la capa y los pesos, de esta manera, se captura la importancia del mapa de activaciones para la clase en particular y se ve reflejado en la imagen de entrada como un mapa de calor con intensidades más altas en aquellas regiones relevantes para la red con las que clasificó la imagen en cierta categoría.[m
[32m+[m[32m- Marlon Valencia — [@marlonvalenciamentor-eng](https://github.com/marlonvalenciamentor-eng)[m
 [m
[31m-## Proyecto original realizado por:[m
[32m+[m[32mProyecto guía del curso *Desarrollo de Proyectos de Inteligencia Artificial*,[m
[32m+[m[32mespecialización en Inteligencia Artificial, Universidad Autónoma de Occidente.[m
 [m
[31m-Isabella Torres Revelo - https://github.com/isa-tr[m
[31m-Nicolas Diaz Salazar - https://github.com/nicolasdiazsalazar[m
[32m+[m[32mRepositorio base: [dalquinones/UAO-Neumonia](https://github.com/dalquinones/UAO-Neumonia)[m
[1mdiff --git a/requirements.txt b/requirements.txt[m
[1mindex eb0b6e3..3db6a4a 100644[m
[1m--- a/requirements.txt[m
[1m+++ b/requirements.txt[m
[36m@@ -1,10 +1,62 @@[m
[31m-pyautogui[m
[31m-pillow[m
[31m-tkcap[m
[31m-pydicom[m
[31m-img2pdf[m
[31m-opencv_python[m
[31m-matplotlib[m
[31m-pandas[m
[31m-tensorflow[m
[31m-python-xlib[m
[32m+[m[32m﻿absl-py==2.5.0[m
[32m+[m[32mastunparse==1.6.3[m
[32m+[m[32mcertifi==2026.7.22[m
[32m+[m[32mcharset-normalizer==3.5.1[m
[32m+[m[32mcolorama==0.4.6[m
[32m+[m[32mcontourpy==1.3.3[m
[32m+[m[32mcycler==0.12.1[m
[32m+[m[32mflatbuffers==25.12.19[m
[32m+[m[32mfonttools==4.64.0[m
[32m+[m[32mgast==0.7.0[m
[32m+[m[32mgoogle-pasta==0.2.0[m
[32m+[m[32mgrpcio==1.83.1[m
[32m+[m[32mh5py==3.14.0[m
[32m+[m[32midna==3.19[m
[32m+[m[32mimg2pdf==0.6.3[m
[32m+[m[32miniconfig==2.3.0[m
[32m+[m[32mkeras==3.15.1[m
[32m+[m[32mkiwisolver==1.5.1[m
[32m+[m[32mlibclang==18.1.1[m
[32m+[m[32mlxml==6.1.3[m
[32m+[m[32mmarkdown-it-py==4.2.0[m
[32m+[m[32mmatplotlib==3.11.1[m
[32m+[m[32mmdurl==0.1.2[m
[32m+[m[32mml-dtypes==0.6.0[m
[32m+[m[32mmouseinfo==0.1.3[m
[32m+[m[32mnamex==0.1.0[m
[32m+[m[32mnumpy==2.5.2[m
[32m+[m[32mopencv-python==5.0.0.93[m
[32m+[m[32mopt-einsum==3.4.0[m
[32m+[m[32moptree==0.20.0[m
[32m+[m[32mpackaging==26.3[m
[32m+[m[32mpandas==3.0.5[m
[32m+[m[32mpikepdf==10.12.0[m
[32m+[m[32mpillow==12.3.0[m
[32m+[m[32mpluggy==1.6.0[m
[32m+[m[32mprotobuf==7.36.1[m
[32m+[m[32mpyautogui==0.9.54[m
[32m+[m[32mpydicom==3.0.2[m
[32m+[m[32mpygetwindow==0.0.9[m
[32m+[m[32mpygments==2.21.0[m
[32m+[m[32mpymsgbox==2.0.1[m
[32m+[m[32mpyparsing==3.3.2[m
[32m+[m[32mpyperclip==1.11.0[m
[32m+[m[32mpyrect==0.2.0[m
[32m+[m[32mpyscreeze==1.0.1[m
[32m+[m[32mpytest==9.1.1[m
[32m+[m[32mpython-dateutil==2.9.0.post0[m
[32m+[m[32mpython-xlib==0.33[m
[32m+[m[32mpytweening==1.2.0[m
[32m+[m[32mrequests==2.34.2[m
[32m+[m[32mrich==15.0.0[m
[32m+[m[32msetuptools==84.0.0[m
[32m+[m[32msix==1.17.0[m
[32m+[m[32mtensorflow==2.21.0[m
[32m+[m[32mtermcolor==3.3.0[m
[32m+[m[32mtf-keras==2.21.0[m
[32m+[m[32mtkcap==0.0.4[m
[32m+[m[32mtyping-extensions==4.16.0[m
[32m+[m[32mtzdata==2026.3[m
[32m+[m[32murllib3==2.7.0[m
[32m+[m[32mwheel==0.48.0[m
[32m+[m[32mwrapt==2.4.0[m
