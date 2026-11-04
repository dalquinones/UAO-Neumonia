"""Preprocesamiento del curso: gris, CLAHE, 512×512 y normalización [0, 1].

El resultado es un tensor NumPy con dimensiones (1, 512, 512, 1).
Se conserva separado de la lectura, la UI y la inferencia.
"""

import cv2
import numpy as np
from PIL import Image

TAMANO = 512


def preprocess_image(imagen: Image.Image | np.ndarray) -> np.ndarray:
    """Transforma una imagen RGB/gris a un tensor de escala de grises normalizado.

    Raises:
        ValueError: si el dato no tiene forma de imagen válida.
    """
    datos = np.asarray(imagen)
    if datos.size == 0 or datos.ndim not in (2, 3):
        raise ValueError("La imagen debe ser una matriz no vacía de 2 o 3 dimensiones")
    if datos.ndim == 3:
        if datos.shape[-1] not in (3, 4):
            raise ValueError("La imagen debe tener 3 o 4 canales")
        rgb = np.asarray(Image.fromarray(datos.astype(np.uint8)).convert("RGB"))
        gris = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        gris = np.asarray(datos, dtype=np.uint8)
    if gris.shape[0] == 0 or gris.shape[1] == 0:
        raise ValueError("Imagen sin dimensiones válidas")
    gris = cv2.resize(gris, (TAMANO, TAMANO), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    corregida = clahe.apply(gris)
    tensor = corregida.astype(np.float32) / 255.0
    return tensor[None, :, :, None]
