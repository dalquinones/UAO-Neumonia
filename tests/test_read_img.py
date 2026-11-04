"""Pruebas de lectura y validación: patrón AAA."""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from src.neumonia.read_img import (
    ImagenInvalida,
    cargar_imagen,
    extension_valida,
    normalizar_pixeles,
)


@pytest.mark.parametrize("nombre", ["imagen.DCM", "radiografia.jpg", "x.jpeg", "x.PNG"])
def test_acepta_formatos_soportados(nombre):
    # Arrange / Act / Assert
    assert extension_valida(nombre)


@pytest.mark.parametrize("nombre", ["archivo.exe", "imagen.svg", "reporte.pdf"])
def test_rechaza_extension_no_soportada(nombre):
    assert not extension_valida(nombre)


def test_lectura_y_visualizacion_jpeg():
    # Arrange
    buffer = BytesIO()
    Image.new("RGB", (20, 10), color=(100, 120, 140)).save(buffer, format="JPEG")
    # Act
    imagen = cargar_imagen(buffer.getvalue(), "radiografia.jpg")
    # Assert
    assert imagen.mode == "RGB"
    assert imagen.size == (20, 10)


def test_lectura_png():
    buffer = BytesIO()
    Image.new("L", (24, 12), color=128).save(buffer, format="PNG")
    salida = cargar_imagen(buffer.getvalue(), "radiografia.png")
    assert salida.size == (24, 12)
    assert salida.mode == "RGB"


def test_rechaza_archivo_vacio():
    with pytest.raises(ImagenInvalida, match="vacío"):
        cargar_imagen(b"", "rayos.png")


def test_rechaza_imagen_corrupta():
    with pytest.raises(ImagenInvalida, match="válida"):
        cargar_imagen(b"no es una imagen", "rayos.png")


def test_normaliza_intensidades_e_invierte_monochrome1():
    matriz = np.array([[0, 100], [200, 300]], dtype=np.int16)
    normal = normalizar_pixeles(matriz)
    invertida = normalizar_pixeles(matriz, invertir=True)
    assert normal.dtype == np.uint8
    assert normal[0, 0] == 0 and normal[-1, -1] == 255
    assert np.array_equal(invertida, 255 - normal)


def test_rechaza_array_dicom_malformado():
    with pytest.raises(ImagenInvalida, match="2D"):
        normalizar_pixeles(np.zeros((10, 10, 3), dtype=np.uint8))


def test_rechaza_dicom_sin_valores_finitos():
    with pytest.raises(ImagenInvalida, match="válidos"):
        normalizar_pixeles(np.full((5, 5), np.nan))
