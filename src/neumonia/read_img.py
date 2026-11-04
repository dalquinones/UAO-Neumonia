"""Lectura y validación de radiografías DICOM/JPEG/PNG."""

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

EXTENSIONES_PERMITIDAS = {".dcm", ".jpeg", ".jpg", ".png"}


class ImagenInvalida(ValueError):
    """El archivo no es una radiografía legible en un formato soportado."""


def extension_valida(nombre: str) -> bool:
    """Valida la extensión, sin usar el nombre como ruta de escritura."""
    return Path(nombre.lower()).suffix in EXTENSIONES_PERMITIDAS


def normalizar_pixeles(pixeles: np.ndarray, invertir: bool = False) -> np.ndarray:
    """Normaliza intensidades arbitrarias a uint8 para visualización."""
    datos = np.asarray(pixeles, dtype=np.float32)
    if datos.ndim != 2 or datos.size == 0:
        raise ImagenInvalida("La radiografía DICOM debe contener una imagen 2D.")
    validos = np.isfinite(datos)
    if not validos.any():
        raise ImagenInvalida("La radiografía no contiene píxeles válidos.")
    minimo, maximo = float(datos[validos].min()), float(datos[validos].max())
    if maximo == minimo:
        salida = np.zeros(datos.shape, dtype=np.uint8)
    else:
        datos = np.nan_to_num(datos, nan=minimo, posinf=maximo, neginf=minimo)
        salida = np.clip((datos - minimo) / (maximo - minimo) * 255, 0, 255).astype(np.uint8)
    return 255 - salida if invertir else salida


def cargar_imagen(contenido: bytes, nombre: str) -> Image.Image:
    """Devuelve una imagen RGB; no persiste datos de pacientes en disco."""
    if not extension_valida(nombre):
        raise ImagenInvalida("Formato no admitido. Usa .dcm, .jpg, .jpeg o .png.")
    if not contenido:
        raise ImagenInvalida("El archivo está vacío.")

    if Path(nombre.lower()).suffix == ".dcm":
        try:
            import pydicom

            ds = pydicom.dcmread(BytesIO(contenido))
            pixeles = ds.pixel_array
            if pixeles.ndim == 3 and pixeles.shape[0] == 1:
                pixeles = pixeles[0]
            matriz = normalizar_pixeles(
                pixeles, invertir=str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1"
            )
            return Image.fromarray(matriz).convert("RGB")
        except ImagenInvalida:
            raise
        except Exception as exc:
            raise ImagenInvalida(f"No se pudo leer el archivo DICOM: {exc}") from exc

    try:
        imagen = Image.open(BytesIO(contenido))
        imagen.load()
        return imagen.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImagenInvalida("El archivo no contiene una imagen JPEG/PNG válida.") from exc
