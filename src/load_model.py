"""
=============================================================================
MÓDULO 3: load_model.py
ARQUITECTURA: Alta Cohesión y Bajo Acoplamiento (ArjanCodes / Clean Code)
RESPONSABILIDAD ÚNICA:
    Carga segura, inspección estructural y validación estricta de la integridad 
    del modelo convolucional de Deep Learning (.h5).

OBJETIVOS ESPECÍFICOS POR FUNCIÓN:

1. validate_model_integrity(model: tf.keras.Model, required_layer: str) -> bool:
   - Objetivo: Certificar que el modelo cargado cumple con la arquitectura clínica requerida.
   - Validaciones:
       a) Comprueba que la entrada acepte tensores de forma (None, 512, 512, 1).
       b) Comprueba que la salida tenga 3 neuronas (Bacteriana, Normal, Viral).
       c) Comprueba que exista la capa convolucional clave para Grad-CAM ('conv10_thisone').
   - Excepciones: Lanza ValueError si alguna dimensión o capa no coincide.

2. load_cnn_model(model_path: str = None) -> tf.keras.Model:
   - Objetivo: Localizar el archivo en disco, cargarlo con Keras 3 y validar su integridad.
   - Estrategia de búsqueda:
       Si no se indica ruta, busca automáticamente 'conv_MLP_84.h5' o 'WilhemNet86.h5'.
   - Excepciones: Lanza FileNotFoundError si el archivo no existe o ValueError si está corrupto.
=============================================================================
"""

import os
import sys
from typing import Any
from functools import lru_cache

# Silenciar mensajes informativos y de advertencia de TensorFlow/CUDA a nivel de C++
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

_stderr_fd = sys.stderr.fileno()
_saved_stderr = os.dup(_stderr_fd)
_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, _stderr_fd)
os.close(_devnull)

try:
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    import tensorflow as tf
finally:
    os.dup2(_saved_stderr, _stderr_fd)
    os.close(_saved_stderr)



def validate_model_integrity(model: Any, required_layer: str = "conv10_thisone") -> bool:
    """
    Verifica que la red neuronal tenga la estructura esperada por el sistema clínico.
    
    Asegura que el modelo acepta la forma de entrada correcta, genera el número
    de clases requeridas y contiene la capa convolucional necesaria para Grad-CAM.
    
    Args:
        model (tf.keras.Model): El modelo de red neuronal a validar.
        required_layer (str, optional): Nombre de la capa convolucional que debe existir.
            Por defecto es "conv10_thisone".
            
    Returns:
        bool: True si la integridad del modelo es válida.
        
    Raises:
        ValueError: Si la forma de entrada, forma de salida o las capas no 
                    cumplen con los requisitos del sistema.
    """
    # 1. Validar forma de entrada esperada: (None, 512, 512, 1)
    input_shape = model.input_shape
    # Si input_shape es una lista (modelos funcionales complejos), tomamos el primer elemento
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
        
    expected_spatial = (512, 512, 1)
    # Comparamos las últimas 3 dimensiones (alto, ancho, canales)
    if input_shape[1:] != expected_spatial:
        raise ValueError(
            f"Integridad fallida en la entrada: Se esperaba {expected_spatial}, "
            f"pero el modelo tiene {input_shape[1:]}"
        )

    # 2. Validar forma de salida esperada: 3 clases
    output_shape = model.output_shape
    if isinstance(output_shape, list):
        output_shape = output_shape[0]

    if output_shape[-1] != 3:
        raise ValueError(
            f"Integridad fallida en la salida: Se esperaban 3 clases (Bacteriana, Normal, Viral), "
            f"pero el modelo tiene {output_shape[-1]}"
        )

    # 3. Validar existencia de la capa convolucional para Grad-CAM
    layer_names = [layer.name for layer in model.layers]
    if required_layer not in layer_names:
        raise ValueError(
            f"Integridad fallida: No se encontró la capa '{required_layer}' requerida para Grad-CAM. "
            f"Capas disponibles: {len(layer_names)}"
        )

    return True


@lru_cache(maxsize=1)
def load_cnn_model(model_path: str = None) -> Any:
    """
    Carga el modelo convolucional (.h5) desde el disco y certifica su integridad.
    
    Si no se proporciona una ruta específica, intentará encontrar automáticamente
    los modelos estándar del proyecto ('conv_MLP_84.h5' o 'WilhemNet86.h5').
    
    Args:
        model_path (str, optional): Ruta explícita al archivo .h5 del modelo.
            Si es None, se buscarán los candidatos por defecto en el directorio local.
            
    Returns:
        tf.keras.Model: Instancia del modelo Keras validado y listo para inferencia.
        
    Raises:
        FileNotFoundError: Si no se encuentra el modelo en la ruta especificada
                           o no hay candidatos por defecto disponibles.
        ValueError: Si el archivo del modelo está vacío (0 bytes) o si falla
                    la validación de integridad arquitectónica.
    """
    # 1. Si no se especificó ruta, buscar los modelos estándar del proyecto
    if model_path is None:
        candidates = ["conv_MLP_84.h5", "WilhemNet86.h5"]
        for candidate in candidates:
            if os.path.exists(candidate):
                model_path = candidate
                break

    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontró ningún archivo de modelo válido. Ruta buscada: '{model_path}'"
        )

    # 2. Validar que el archivo no esté vacío (0 bytes)
    if os.path.getsize(model_path) == 0:
        raise ValueError(f"El archivo del modelo '{model_path}' está vacío o corrupto (0 bytes).")

    # 3. Cargar el modelo con tf_keras explícito (evita bugs de lazy loading en pytest con TF > 2.15)
    import tf_keras
    model = tf_keras.models.load_model(model_path, compile=False)

    # 4. Validar la integridad del modelo
    validate_model_integrity(model)

    return model


# Función de compatibilidad hacia atrás
def model_fun() -> Any:
    """
    Función de conveniencia compatible con el código legacy.
    
    Envuelve 'load_cnn_model' para no romper integraciones existentes que 
    dependen de la firma antigua.
    
    Returns:
        tf.keras.Model: El modelo clínico cargado y validado.
    """
    return load_cnn_model()
