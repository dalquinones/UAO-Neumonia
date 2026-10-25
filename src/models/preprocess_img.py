"""Capa modelo: preprocesamiento de radiografías para inferencia.

Responsabilidad única: transformar UNA imagen (array BGR de cualquier tamaño)
en el tensor exacto que espera la red: (1, 512, 512, 1) flotante en [0, 1].

El orden y los parámetros replican el pipeline de ENTRENAMIENTO del modelo:
cambiarlos aquí desincroniza la inferencia con lo que la red aprendió.
"""
import cv2
import numpy as np

TARGET_SIZE = (512, 512)          # tamaño de entrada que exige la topología
CLAHE_CLIP_LIMIT = 2.0            # límite de realce de contraste local
CLAHE_TILE_GRID = (4, 4)          # grilla de regiones del histograma adaptativo


def preprocess(array: np.ndarray) -> np.ndarray:
    """Transforma la imagen al tensor de entrada de la red.

    Pipeline: resize 512x512 → escala de grises → CLAHE → /255 → batch.

    Args:
        array: imagen BGR uint8 de cualquier tamaño (salida de read_img)

    Returns:
        np.ndarray float32 con forma (1, 512, 512, 1), valores en [0, 1]

    Raises:
        ValueError: si el array está vacío o no tiene 1 ni 3 canales.
    """
    if array is None or array.size == 0:
        raise ValueError("El array de entrada está vacío")

    # ① homogeneiza el canal de entrada (gris 2D → BGR 3 canales)
    canales = 1 if array.ndim == 2 else array.shape[-1]
    if canales == 1:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    elif canales != 3:
        raise ValueError(f"Se esperaban 1 o 3 canales (gris/BGR); recibí {canales}")

    # ② resize al tamaño exacto que pide la topología de la red
    resized = cv2.resize(array, TARGET_SIZE)

    # ③ conversión a escala de grises ponderada (idéntica al entrenamiento)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # ④ CLAHE: realce adaptativo del contraste por regiones 4x4
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    enhanced = clahe.apply(gray)

    # ⑤ normalización a [0, 1] — la red fue entrenada con ese rango
    normalized = enhanced.astype(np.float32) / 255.0

    # ⑥ formato tensor: (h, w, 1) → batch (1, h, w, 1)
    tensor = normalized[:, :, np.newaxis]
    return tensor[np.newaxis, :, :, :]