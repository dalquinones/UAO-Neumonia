"""
=============================================================================
MÓDULO 2: preprocess_img.py
ARQUITECTURA: Alta Cohesión y Bajo Acoplamiento (ArjanCodes / Clean Code)
RESPONSABILIDAD ÚNICA:
    Transformación morfológica, ecualización adaptativa de histograma y
    normalización numérica de matrices de imagen para la red convolucional.

OBJETIVO ESPECÍFICO DE LA FUNCIÓN:
preprocess_image(array: np.ndarray) -> np.ndarray:
    1. Valida el tipo y dimensiones del arreglo de entrada.
    2. Redimensiona espacialmente a 512x512 píxeles.
    3. Convierte a escala de grises de 1 solo canal (si viene en 3 canales).
    4. Aplica ecualización CLAHE (Contrast Limited Adaptive Histogram Equalization)
       para resaltar estructuras óseas y opacidades pulmonares.
    5. Normaliza los valores de intensidad al rango flotante [0.0, 1.0].
    6. Expande dimensiones a formato de batch (Tensor) de 4D: (1, 512, 512, 1).

ENTREGA:
    Tensor NumPy de 4 dimensiones (float32) con forma (1, 512, 512, 1).
=============================================================================
"""

import cv2
import numpy as np


def preprocess_image(array: np.ndarray) -> np.ndarray:
    """
    Preprocesa una imagen radiográfica (NumPy array) para la entrada de la CNN.
    
    Aplica redimensionamiento espacial a 512x512, conversión a escala de grises,
    ecualización adaptativa de contraste local (CLAHE), normalización al rango [0.0, 1.0]
    y expansión de dimensiones a formato de batch de Keras/TensorFlow.
    
    Args:
        array (np.ndarray): Matriz numérica de la imagen original a preprocesar.
        
    Returns:
        np.ndarray: Tensor 4D de coma flotante de forma (1, 512, 512, 1) normalizado entre 0.0 y 1.0.
        
    Raises:
        TypeError: Si el parámetro 'array' no es una instancia de np.ndarray.
        ValueError: Si el arreglo está vacío, no tiene dimensiones válidas, 
                    o tiene un número de canales no soportado.
    """
    # 1. Validación de tipo de dato
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Se esperaba un numpy.ndarray, pero se recibió: {type(array)}")

    # 2. Validación de dimensiones
    if array.size == 0 or len(array.shape) < 2:
        raise ValueError("El arreglo de entrada está vacío o no tiene dimensiones de imagen válidas.")

    # Asegurar compatibilidad de tipo de dato con OpenCV resize (CV_8U, CV_16U, CV_32F, CV_64F)
    if array.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        array = array.astype(np.float32)

    # 3. Redimensionamiento espacial estándar (512 x 512)
    resized = cv2.resize(array, (512, 512), interpolation=cv2.INTER_AREA)

    # 4. Conversión a escala de grises (1 canal)
    if len(resized.shape) == 3:
        if resized.shape[2] == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        elif resized.shape[2] == 1:
            gray = resized[:, :, 0]
        else:
            raise ValueError(f"Número de canales no soportado: {resized.shape[2]}")
    else:
        gray = resized

    # Asegurar que los datos para CLAHE sean enteros de 8 bits (0 - 255)
    if gray.dtype != np.uint8:
        min_val = gray.min()
        max_val = gray.max()
        if max_val > min_val:
            gray = np.uint8(((gray - min_val) / (max_val - min_val)) * 255.0)
        else:
            if gray.dtype == np.uint16:
                gray = np.uint8((gray / 65535.0) * 255.0)
            elif gray.dtype in (np.float32, np.float64) and 0.0 <= max_val <= 1.0:
                gray = np.uint8(gray * 255.0)
            else:
                raise ValueError("La imagen carece de contraste y está fuera de rangos fiables. No es posible normalizarla de forma segura.")

    # 5. Ecualización adaptativa de contraste local (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    equalized = clahe.apply(gray)

    # 6. Normalización flotante en el rango [0.0, 1.0]
    normalized = equalized.astype(np.float32) / 255.0

    # 7. Expansión a 4 dimensiones para formato de batch de Keras/TensorFlow: (1, 512, 512, 1)
    batch_tensor = np.expand_dims(normalized, axis=-1)  # (512, 512, 1)
    batch_tensor = np.expand_dims(batch_tensor, axis=0)  # (1, 512, 512, 1)

    return batch_tensor
