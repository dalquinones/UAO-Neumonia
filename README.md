## Hola! Bienvenido a la herramienta para la detección rápida de neumonía

Deep Learning aplicado en el procesamiento de imágenes radiográficas de tórax en formato DICOM con el fin de clasificarlas en 3 categorías diferentes:

1. Neumonía Bacteriana

2. Neumonía Viral

3. Sin Neumonía

Aplicación de una técnica de explicación llamada Grad-CAM para resaltar con un mapa de calor las regiones relevantes de la imagen de entrada.

---

## Uso de la herramienta:

A continuación le explicaremos cómo empezar a utilizarla.

## Estado del repositorio (en refactor)

- Migración MVC completa: `src/models` + `src/controllers` verificados (recepción,
  pytest, ruff y py_compile en verde); vista `src/views` a cargo de un integrante.
- **Fuente única de verdad:** [`CONSTITUTION.md`](CONSTITUTION.md) · contrato de
  equipo: [`CONVENCIONES.md`](CONVENCIONES.md).
- Rama activa: `feature/refactor-mvc` → PRs hacia `develop` (revisa el code reviewer).

---

## Uso de la herramienta (todo con `uv`)

> Regla dura del proyecto: la gestión de dependencias y entornos pasa por
> [`uv`](https://docs.astral.sh/uv/). `pip`, `conda` y `requirements.txt` están
> prohibidos.

1. Instale `uv` si no lo tiene:

   ```bash
   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Clone el repositorio y entre a la carpeta:

   ```bash
   git clone https://github.com/JuanC0319/UAO-Neumonia.git
   cd UAO-Neumonia
   ```

3. Cree el entorno determinista desde `pyproject.toml` + `uv.lock`:

   ```bash
   uv sync
   ```

4. Copie el modelo entrenado `WilhemNet_86.h5` en la raíz del proyecto (no se versiona
   por peso; lo comparte el equipo por unidad externa).

5. Verifique que el entorno está sano:

   ```bash
   uv run python verify_pipeline.py
   ```

> Nota: en macOS Intel TensorFlow no tiene rueda oficial y vuela fuera del entorno
> local. El modelo se carga de forma diferida (patrón TF-lazy) y TensorFlow real solo
> entra en juego dentro de Docker/CI.

Uso de la Interfaz Gráfica:

> La interfaz se rehace en `src/views/` sobre el `Integrator` (decisión Tkinter vs
> Streamlit pendiente del equipo). El flujo de la UI legada (cédula, 'Cargar Imagen',
> 'Predecir', 'Guardar' CSV, 'PDF', 'Borrar') es el contrato funcional que la nueva
> vista debe replicar.

---

## Arquitectura (MVC)

```text
src/
├── models/                # capa modelo — aquí vive TODO lo pesado
│   ├── read_img.py        #   DICOM/JPG → (array BGR uint8, preview PIL)
│   ├── preprocess_img.py  #   tensor (1, 512, 512, 1) float32 [0-1] con CLAHE
│   ├── load_model.py      #   carga lazy de TF + validación de WilhemNet_86
│   └── grad_cam.py        #   heatmap Grad-CAM + overlay (explicabilidad)
├── controllers/
│   └── integrator.py      # fachada única con modelo INYECTADO (DI)
└── views/                 # interfaz gráfica (decisión Tkinter vs Streamlit en curso)
```

Reglas MVC (detalle en [`CONSTITUTION.md`](CONSTITUTION.md) §5):

- La vista NUNCA importa `tensorflow`/`cv2`/`pydicom`: solo habla con `Integrator`.
- Solo `load_model.py` conoce TensorFlow y lo importa de forma diferida (*lazy*).
- El modelo entra por inyección: producción = Keras real; pruebas = `FakeModel`
  (dobles en `test/conftest.py`) — así los 120 tests AAA corren sin TF en local.

---

## Acerca del Modelo

La red neuronal convolucional implementada (CNN) es basada en el modelo implementado por F. Pasa, V.Golkov, F. Pfeifer, D. Cremers & D. Pfeifer
en su artículo Efcient Deep Network Architectures for Fast Chest X-Ray Tuberculosis Screening and Visualization.

Está compuesta por 5 bloques convolucionales, cada uno contiene 3 convoluciones; dos secuenciales y una conexión 'skip' que evita el desvanecimiento del gradiente a medida que se avanza en profundidad.
Con 16, 32, 48, 64 y 80 filtros de 3x3 para cada bloque respectivamente.

Después de cada bloque convolucional se encuentra una capa de max pooling y después de la última una capa de Average Pooling seguida por tres capas fully-connected (Dense) de 1024, 1024 y 3 neuronas respectivamente.

Para regularizar el modelo utilizamos 3 capas de Dropout al 20%; dos en los bloques 4 y 5 conv y otra después de la 1ra capa Dense.

## Acerca de Grad-CAM

Es una técnica utilizada para resaltar las regiones de una imagen que son importantes para la clasificación. Un mapeo de activaciones de clase para una categoría en particular indica las regiones de imagen relevantes utilizadas por la CNN para identificar esa categoría.

Grad-CAM realiza el cálculo del gradiente de la salida correspondiente a la clase a visualizar con respecto a las neuronas de una cierta capa de la CNN. Esto permite tener información de la importancia de cada neurona en el proceso de decisión de esa clase en particular. Una vez obtenidos estos pesos, se realiza una combinación lineal entre el mapa de activaciones de la capa y los pesos, de esta manera, se captura la importancia del mapa de activaciones para la clase en particular y se ve reflejado en la imagen de entrada como un mapa de calor con intensidades más altas en aquellas regiones relevantes para la red con las que clasificó la imagen en cierta categoría.

## Proyecto original realizado por:

Isabella Torres Revelo - https://github.com/isa-tr
Nicolas Diaz Salazar - https://github.com/nicolasdiazsalazar
