"""
=============================================================================
MÓDULO 4: grad_cam.py
ARQUITECTURA: Alta Cohesión y Bajo Acoplamiento (ArjanCodes / Clean Code)
RESPONSABILIDAD ÚNICA:
    Inferencia del modelo convolucional y generación de explicabilidad visual
    mediante la técnica Grad-CAM (Gradient-weighted Class Activation Mapping).

OBJETIVO ESPECÍFICO:
    Integrar la red neuronal con la imagen de entrada para obtener:
      1. Clase diagnóstica predicha: 'bacteriana', 'normal' o 'viral'.
      2. Probabilidad de certeza médica (porcentaje: 0.0% a 100.0%).
      3. Mapa de calor (Heatmap) superpuesto sobre la radiografía original.
=============================================================================
"""

import os
import sys
import cv2
import numpy as np
from typing import Any

# Silenciar mensajes informativos y de advertencia de TensorFlow/CUDA a nivel de C++
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

_stderr_fd = sys.stderr.fileno()
_saved_stderr = os.dup(_stderr_fd)
_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, _stderr_fd)
os.close(_devnull)

try:
    import tensorflow as tf
finally:
    os.dup2(_saved_stderr, _stderr_fd)
    os.close(_saved_stderr)


from src.load_model import load_cnn_model
from src.preprocess_img import preprocess_image

# Diccionario clínico oficial de etiquetas diagnósticas
DIAGNOSTIC_LABELS = {
    0: "bacteriana",
    1: "normal",
    2: "viral"
}


def generate_gradcam_heatmap(
    model: Any,
    batch_tensor: np.ndarray,
    layer_name: str = "conv10_thisone"
) -> np.ndarray:
    """
    Calcula el mapa de activación de clase ponderado por gradientes (Grad-CAM).
    
    Args:
        model (tf.keras.Model): Modelo de Keras cargado y validado.
        batch_tensor (np.ndarray): Tensor de entrada preprocesado de forma (1, 512, 512, 1).
        layer_name (str, optional): Nombre de la capa convolucional de la cual extraer 
                                    los mapas de activación. Por defecto es "conv10_thisone".
                                    
    Returns:
        np.ndarray: Matriz 2D de intensidades espaciales (512x512) normalizada entre 0.0 y 1.0.
    """
    # 1. Crear sub-modelo extractor que devuelva la capa convolucional y la predicción final
    last_conv_layer = model.get_layer(layer_name)
    import tf_keras
    grad_model = tf_keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    # 2. Registrar las operaciones con GradientTape en modo Eager
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([batch_tensor])
        # Asegurar extracción de tensor si Keras 3 devuelve lista
        if isinstance(predictions, list):
            predictions = predictions[0]
        if isinstance(conv_outputs, list):
            conv_outputs = conv_outputs[0]

        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # 3. Gradiente de la clase seleccionada respecto a la salida convolucional
    grads = tape.gradient(class_channel, conv_outputs)

    # 4. Promedio espacial de los gradientes (Global Average Pooling)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_outputs = conv_outputs[0].numpy()

    # 5. Ponderar cada canal convolucional por la importancia de su gradiente
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    # 6. Promediar canales y aplicar ReLU (solo resaltar activaciones positivas)
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    # 7. Redimensionar al tamaño estándar de 512 x 512
    heatmap = cv2.resize(heatmap, (512, 512))
    return heatmap


def superimpose_heatmap(
    original_array: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.8
) -> np.ndarray:
    """
    Superpone el mapa de calor con mapa de color JET sobre la imagen original.
    
    Args:
        original_array (np.ndarray): Matriz numérica de la imagen radiográfica original.
        heatmap (np.ndarray): Mapa de calor Grad-CAM 2D normalizado [0.0, 1.0].
        alpha (float, optional): Opacidad del mapa de calor superpuesto. Por defecto 0.8.
        
    Returns:
        np.ndarray: Imagen RGB (uint8) de 512x512 con el mapa de calor visualizable.
    """
    # Convertir el mapa de calor a formato de color JET (8 bits)
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Adaptar la radiografía original a 512x512 y 3 canales BGR para superposición
    base_img = cv2.resize(original_array, (512, 512))
    if len(base_img.shape) == 2:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    elif base_img.shape[2] == 1:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)

    # Mezcla ponderada de la radiografía y el mapa de calor
    transparency = (colored_heatmap * alpha).astype(np.uint8)
    superimposed = cv2.add(transparency, base_img).astype(np.uint8)

    # Convertir a RGB para que sea compatible con Pillow y la interfaz
    return superimposed[:, :, ::-1]


def predict_and_explain(
    original_array: np.ndarray,
    model: Any = None,
    layer_name: str = "conv10_thisone"
) -> tuple[str, float, np.ndarray]:
    """
    Función principal del módulo: Realiza la clasificación clínica y genera explicabilidad Grad-CAM.
    
    Args:
        original_array (np.ndarray): Matriz numérica de la imagen radiográfica original.
        model (tf.keras.Model, optional): Modelo pre-cargado. Si es None, se cargará desde el disco.
        layer_name (str, optional): Nombre de la capa convolucional para Grad-CAM.
        
    Returns:
        tuple[str, float, np.ndarray]: Una tupla con:
            - str: Clase médica predicha ('bacteriana', 'normal', 'viral').
            - float: Probabilidad de certeza (0.0% a 100.0%).
            - np.ndarray: Imagen con el mapa de calor Grad-CAM superpuesto.
    """
    # 1. Cargar modelo si no fue suministrado
    if model is None:
        model = load_cnn_model()

    # 2. Preprocesar la imagen
    batch_tensor = preprocess_image(original_array)

    # 3. Inferencia de probabilidades
    predictions = model.predict(batch_tensor, verbose=0)
    if isinstance(predictions, list):
        predictions = predictions[0]

    predicted_idx = int(np.argmax(predictions[0]))
    probability = float(np.max(predictions[0]) * 100.0)
    label = DIAGNOSTIC_LABELS.get(predicted_idx, "desconocido")

    # 4. Generar mapa de calor Grad-CAM
    heatmap_raw = generate_gradcam_heatmap(model, batch_tensor, layer_name)
    superimposed_img = superimpose_heatmap(original_array, heatmap_raw)

    return label, probability, superimposed_img


# Compatibilidad hacia atrás con el código legacy
def grad_cam(array: np.ndarray) -> np.ndarray:
    """
    Función legacy compatible: retorna directamente la imagen superpuesta.
    
    Args:
        array (np.ndarray): Matriz numérica de la radiografía original.
        
    Returns:
        np.ndarray: La imagen RGB (uint8) procesada con el mapa de calor superpuesto.
    """
    _, _, superimposed_img = predict_and_explain(array)
    return superimposed_img
