"""Capa modelo: explicabilidad con Grad-CAM (Class Activation Mapping).

Responsabilidad única: generar el mapa de calor que muestra qué zonas
de la radiografía favorecieron la predicción, y componerlo sobre la
imagen en colores.

Pipeline: gradientes de la clase ganadora respecto a la última conv
→ importancia media por canal → suma ponderada de activaciones.
En TF2 los gradientes se calculan con tf.GradientTape (el monolito
usaba K.gradients, API de TF1 eliminada — bug corregido del refactor).
"""
import cv2
import numpy as np

from src.models.load_model import CONV_LAYER_NAME  # constante compartida

DEFAULT_ALPHA = 0.4                                 # peso del mapa en la mezcla


def make_gradcam_heatmap(modelo, tensor, class_index=None):
    """Genera el mapa de calor 2D de una clase sobre el modelo.

    Args:
        modelo: red Keras (o stub con interfaz equivalente en pruebas)
        tensor: batch (1, H, W, 1) igual al de predict
        class_index: clase objetivo; si None usa la ganadora (argmax)

    Returns:
        np.ndarray float32 con forma (h, w) y valores en [0, 1]

    Raises:
        ValueError: si el modelo no expone la capa conv requerida.
    """
    import tensorflow as tf  # lazy: misma convención de load_model

    try:
        conv_layer = modelo.get_layer(CONV_LAYER_NAME)
    except ValueError as exc:
        raise ValueError(
            f"Grad-CAM necesita la capa '{CONV_LAYER_NAME}' y el modelo no la "
            f"expone. ¿Es el .h5 correcto?"
        ) from exc

    # subgrafo: entrada → [activaciones conv, salida de clases]
    grad_model = tf.keras.Model(modelo.inputs, [conv_layer.output, modelo.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(tensor)
        indice = int(np.argmax(predictions[0])) if class_index is None else class_index
        loss = predictions[:, indice]                     # "qué tan segura es la clase"
    grads = tape.gradient(loss, conv_output)              # dClase / dActivaciones
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # ① importancia por canal
    mapa = conv_output[0] @ pooled_grads[..., tf.newaxis]  # ② activaciones·importancia
    mapa = tf.maximum(mapa, 0.0)                            # ③ ReLU: solo aporta lo que SUMA
    maximo = tf.math.reduce_max(mapa)
    if tf.equal(maximo, 0.0):                               # caso borde: efecto nulo
        return tf.zeros_like(mapa).numpy()
    return (mapa / maximo).numpy()                          # normalizado a [0, 1].


def overlay_heatmap(heatmap: np.ndarray, array: np.ndarray,
                    alpha: float = DEFAULT_ALPHA) -> np.ndarray:
    """Compone el mapa semitransparente sobre la imagen original.

    Args:
        heatmap: (h, w) float en [0, 1] (salida de make_gradcam_heatmap)
        array: imagen BGR uint8 (la misma del read_img)
        alpha: peso del mapa en la mezcla (0 = original, 1 = solo mapa)

    Returns:
        np.ndarray uint8 BGR, misma shape que `array`

    Raises:
        ValueError: si heatmap() no es 2D, array (h,w,3), o alpha fuera [0,1].
    """
    if heatmap.ndim != 2:
        raise ValueError(f"El mapa debe ser 2D (h, w); recibí ndim={heatmap.ndim}")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"La imagen debe ser BGR (h, w, 3); recibí {array.shape}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha debe estar en [0, 1]; recibí {alpha}")

    calor = np.uint8(255.0 * heatmap)                       # 0-255 para OpenCV
    calor = cv2.resize(calor, (array.shape[1], array.shape[0]))
    colores = cv2.applyColorMap(calor, cv2.COLORMAP_JET)    # azul → rojo
    return cv2.addWeighted(array, 1.0 - alpha, colores, alpha, 0)