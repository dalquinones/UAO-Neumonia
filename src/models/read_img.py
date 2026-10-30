"""Capa modelo: lectura y validación de imágenes de entrada (DICOM y JPG/JPEG/PNG).

Responsabilidad única: convertir un archivo en disco a una tupla:
    (array, preview)
Donde:
    array   → ndarray uint8 en formato BGR, listo para preprocess_img
    preview → PIL.Image en RGB, lista para pintarse en la interfaz

Es también la ÚNICA fuente de verdad de los formatos admitidos
(SUPPORTED_EXTENSIONS) y de qué lector usar para cada uno (read_file): sumar un
formato nuevo (p. ej. NIfTI) solo toca este módulo.

Este módulo NO sabe nada de redes neuronales ni de la interfaz.
"""

from pathlib import Path

import cv2
import numpy as np
import pydicom
from PIL import Image

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".dcm", ".jpg", ".jpeg", ".png")


def _check_exists(path: str) -> None:
    """Valida que el archivo exista; si no, lanza FileNotFoundError."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")


def read_dicom(path: str) -> tuple[np.ndarray, Image.Image]:
    """Lee una radiografía en formato DICOM.

    Args:
        path: ruta del archivo .dcm

    Returns:
        (array BGR uint8 0-255, imagen PIL en escala de grises)

    Raises:
        FileNotFoundError: si la ruta no existe.
    """
    _check_exists(path)
    dcm = pydicom.dcmread(path)  # ① parser DICOM
    pixel = dcm.pixel_array.astype(float)  # ② copia como float
    maximo = pixel.max()
    if maximo == 0:  # ③ defensa de caso borde
        gray = np.zeros_like(pixel, dtype=np.uint8)
    else:
        gray = np.uint8((np.maximum(pixel, 0) / maximo) * 255.0)  # ④ normaliza
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # ⑤ unifica convención
    preview = Image.fromarray(gray)  # ⑥ la que pinta la UI
    return bgr, preview


def read_image(path: str) -> tuple[np.ndarray, Image.Image]:
    """Lee una imagen JPG/JPEG/PNG común.

    Args:
        path: ruta del archivo

    Returns:
        (array BGR uint8 tal como la leyó OpenCV, preview PIL en RGB)

    Raises:
        FileNotFoundError: si la ruta no existe o no se puede leer.
    """
    _check_exists(path)
    img = cv2.imread(path)  # ⑦ OpenCV lee en BGR
    if img is None:  # ⑧ ruta corrupta/admitida errada
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    preview = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img, preview


def read_file(path: str) -> tuple[np.ndarray, Image.Image]:
    """Lee cualquier formato soportado eligiendo el lector según la extensión.

    Args:
        path: ruta del archivo (.dcm, .jpg, .jpeg, .png)

    Returns:
        (array BGR uint8, preview PIL) del lector correspondiente.

    Raises:
        FileNotFoundError: si la ruta no existe, es un directorio o no se lee.
    """
    ruta = Path(path)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    if ruta.suffix.lower() == ".dcm":
        return read_dicom(str(ruta))
    return read_image(str(ruta))
