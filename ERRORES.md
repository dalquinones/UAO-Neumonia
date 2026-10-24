# Bitácora de errores y decisiones técnicas

Registro del proceso de puesta en marcha del código base del proyecto guía
**UAO-Neumonia** sobre un entorno Windows con Python 3.13.

El repositorio original fue escrito para **Python 3.8, TensorFlow 2.8, Pillow 9 y
pydicom 2.3**. El curso exige desarrollar en **Python 3.13**, lo que instala
TensorFlow 2.21, Keras 3.15, Pillow 12, pydicom 3.0 y NumPy 2.5. La mayoría de
los fallos encontrados provienen de APIs que fueron eliminadas en ese salto de
versiones, no de errores lógicos del algoritmo.

---

## Resumen

| # | Error | Categoría | Estado |
|---|---|---|---|
| 1 | `ModuleNotFoundError: No module named 'tkinter.tix'` | Entorno | Resuelto |
| 2 | `NameError: name 'tf' is not defined` | Import faltante | Resuelto |
| 3 | `NameError: name 'model_fun' is not defined` | Función faltante | Resuelto |
| 4 | `AttributeError: module 'pydicom' has no attribute 'read_file'` | API eliminada | Resuelto |
| 5 | `NameError: name 'dicom' is not defined` | Import faltante | Resuelto |
| 6 | `NameError: name 'K' is not defined` | Import faltante | Resuelto |
| 7 | `AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'` | API eliminada | Resuelto |
| 8 | El modelo se carga dos veces por predicción | Rendimiento | Resuelto |

---

## Error 1 — `tkinter.tix` no existe en Python 3.13

**Síntoma**

```text
File ".venv\Lib\site-packages\tkcap\__init__.py", line 31, in <module>
    import tkinter.tix as tix
ModuleNotFoundError: No module named 'tkinter.tix'
```

**Causa**

El módulo `tkinter.tix` está deprecado desde Python 3.6 y fue **eliminado en
Python 3.13**. La librería `tkcap` 0.0.4 —requerida para generar el PDF del
reporte— todavía lo importa en su inicialización.

**Alternativas evaluadas**

| Opción | Descartada porque |
|---|---|
| Bajar a Python 3.12 | El curso exige 3.13 explícitamente |
| Reemplazar `tkcap` por otra librería | El profesor pide `tkcap` para el entregable en PDF |
| Editar el paquete en `site-packages` | El cambio se pierde al reinstalar y no viaja en el repositorio |

**Solución**

Módulo `compat_tix.py` en la raíz del proyecto, que registra un `tkinter.tix`
sintético que delega en `tkinter`. Se importa antes que `tkcap`.

```python
import sys
import tkinter
import types

if "tkinter.tix" not in sys.modules:
    _tix = types.ModuleType("tkinter.tix")
    _tix.__getattr__ = lambda name: getattr(tkinter, name)
    sys.modules["tkinter.tix"] = _tix
    tkinter.tix = _tix
```

Es una solución versionada, reversible y que no toca las dependencias instaladas.

---

## Error 2 — Falta `import tensorflow as tf`

**Síntoma**

```text
File "detector_neumonia.py", line 17, in <module>
    tf.compat.v1.disable_eager_execution()
NameError: name 'tf' is not defined
```

**Causa**

El script invocaba `tf.compat.v1.disable_eager_execution()` sin haber importado
TensorFlow en ningún punto del archivo.

**Solución**

Se agregó `import tensorflow as tf` antes de su primer uso. Posteriormente, al
modularizar, la dependencia de TensorFlow salió por completo de la interfaz
gráfica y quedó confinada a `src/load_model.py` y `src/grad_cam.py`.

---

## Error 3 — `model_fun()` no estaba definida

**Síntoma**

```text
NameError: name 'model_fun' is not defined
```

**Causa**

Tanto `predict()` como `grad_cam()` llamaban a `model_fun()`, pero la función no
existía en el archivo. La única referencia a la carga del modelo era una línea
comentada:

```python
# model_cnn = tf.keras.models.load_model('conv_MLP_84.h5')
```

**Solución**

Se implementó la función y, en la fase de modularización, se trasladó a
`src/load_model.py` con validación de existencia del archivo y caché.

---

## Error 4 — `pydicom.read_file` fue eliminado

**Síntoma**

```text
AttributeError: module 'pydicom' has no attribute 'read_file'
```

**Causa**

`read_file` estaba deprecado desde pydicom 2.x y fue **eliminado en pydicom 3.0**.
El entorno instala 3.0.2.

**Solución**

Reemplazo por `dcmread`, que es el nombre vigente de la misma función.

```python
dataset = pydicom.dcmread(path)
```

---

## Error 5 — Falta `import pydicom`

**Síntoma**

```text
NameError: name 'dicom' is not defined
```

**Causa**

`read_dicom_file()` usaba el alias `dicom`, pero el archivo no contenía ningún
`import pydicom as dicom`.

**Solución**

Import agregado. Tras la modularización quedó dentro de `src/read_img.py`, que es
el único módulo que necesita conocer el formato DICOM.

---

## Error 6 — Falta el backend de Keras (`K`)

**Síntoma**

```text
NameError: name 'K' is not defined
```

**Causa**

La implementación original de Grad-CAM usaba `K.gradients`, `K.mean` y
`K.function` sin importar el backend.

**Solución**

En lugar de agregar el import, se resolvió de raíz: la función se reescribió con
`tf.GradientTape`, que es el API vigente. Ver la sección *Decisiones técnicas*.

---

## Error 7 — `Image.ANTIALIAS` fue eliminado

**Síntoma**

```text
File "detector_neumonia.py", line 209, in load_img_file
    self.img1 = img2show.resize((250, 250), Image.ANTIALIAS)
AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'
```

**Causa**

La constante fue deprecada en Pillow 9.1 y **eliminada en Pillow 10**. El entorno
instala Pillow 12.3.

**Solución**

Reemplazo por `Image.Resampling.LANCZOS`, que es exactamente el mismo filtro de
remuestreo con la nomenclatura actual.

---

## Error 8 — El modelo se cargaba dos veces por predicción

**Síntoma**

No producía excepción, pero cada predicción tardaba cerca de 40 segundos. En los
registros de TensorFlow aparecían cuatro construcciones del grafo tras dos
predicciones consecutivas:

```text
Operation '{name:'fc3/Softmax' ...
Operation '{name:'fc3_1/Softmax' ...
Operation '{name:'fc3_2/Softmax' ...
Operation '{name:'fc3_3/Softmax' ...
```

Los sufijos `_1`, `_2`, `_3` indican que Keras registró cuatro instancias
distintas del mismo modelo.

**Causa**

`predict()` llamaba `model_fun()` y luego invocaba `grad_cam()`, que volvía a
llamar `model_fun()` por su cuenta. Cada llamada leía los 112 MB del archivo
`.h5` desde disco. Además, `predict()` ejecutaba `model.predict()` **dos veces**
sobre el mismo lote: una para la clase y otra para la probabilidad.

**Solución**

- `src/load_model.py` usa `functools.lru_cache` para leer el archivo una sola vez.
- `src/integrator.py` carga el modelo y lo **inyecta** en `grad_cam()`.
- Una sola llamada a `model.predict()` alimenta tanto la clase como la
  probabilidad.

```python
predicciones = modelo.predict(lote, verbose=0)
indice = int(np.argmax(predicciones))
probabilidad = float(np.max(predicciones)) * 100
```

---

## Decisiones técnicas

### Keras 3 no puede leer los pesos entregados

TensorFlow 2.21 instala Keras 3.15 por defecto. Los archivos `conv_MLP_84.h5` y
`WilhemNet86.h5` fueron guardados con Keras 2 y no son legibles por Keras 3.

Se instaló `tf-keras` y se activa la variable de entorno antes de importar
TensorFlow:

```python
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
```

La alternativa —convertir los pesos al formato nativo `.keras`— se descartó
porque alteraría el modelo entregado por el docente.

### Grad-CAM con `GradientTape` en lugar del modo grafo

La implementación original requería `tf.compat.v1.disable_eager_execution()` para
que `K.gradients` funcionara. Eso forzaba el modo grafo de TensorFlow 1 y
generaba una docena de advertencias de deprecación en cada ejecución.

La reescritura con `tf.GradientTape` elimina esa dependencia, funciona en modo
eager y reduce las advertencias a tres, todas emitidas por dependencias internas
de TensorFlow (`tf-keras` y `gast`) y ninguna originada en el código del proyecto.

### `python-xlib` se conserva pese a no funcionar en Windows

Es una librería del sistema gráfico X11 de Linux. En Windows se instala pero no
tiene ninguna función. Se mantiene en `requirements.txt` porque `pyautogui` la
necesita en Linux, que es el sistema base del contenedor Docker definido en el
`Dockerfile`.

### El nombre de la capa convolucional es un parámetro

`"conv10_thisone"` estaba incrustado en el cuerpo de `grad_cam()`. Como el
proyecto contempla dos modelos distintos (`conv_MLP_84.h5`)
cuyas capas no tienen por qué llamarse igual, el nombre pasó a ser un parámetro
con valor por defecto.

### Validación de división por cero

La normalización original hacía `(np.maximum(img, 0) / img.max()) * 255.0` sin
verificar el denominador. Una imagen completamente negra producía una división
por cero. Ahora se devuelve un arreglo de ceros en ese caso.

---

## Verificación de portabilidad

Para comprobar que la suite de pruebas no depende de archivos locales, se ejecutó
con el modelo ausente:

```powershell
Rename-Item conv_MLP_84.h5 conv_MLP_84.h5.bak
uv run pytest -v
Rename-Item conv_MLP_84.h5.bak conv_MLP_84.h5
```

Resultado: **10 pasadas, 10 omitidas, 0 fallidas**. Las pruebas que requieren el
modelo se omiten mediante el decorador `hay_modelo` en lugar de fallar, de modo
que cualquier integrante del equipo puede ejecutar la suite sin descargar los
112 MB de pesos.

---

## Entorno final verificado

| Componente | Versión |
|---|---|
| Python | 3.13.3 |
| uv | 0.12.9 |
| TensorFlow | 2.21.0 |
| tf-keras | 2.21.0 |
| Keras | 3.15.1 |
| NumPy | 2.5.2 |
| OpenCV | 5.0.0.93 |
| pydicom | 3.0.2 |
| Pillow | 12.3.0 |
| pytest | 9.1.1 |
| GNU Make | 4.4.1 (instalado con Chocolatey 2.7.4) |

Sistema operativo: Windows. El versionado exacto de todas las dependencias queda
registrado en `requirements.txt` y en `uv.lock`.
