"""Pruebas del modulo de lectura de imagenes."""

import numpy as np
import pytest

from src.read_img import read_dicom_file, read_jpg_file


def test_dicom_devuelve_rgb_y_pil(archivo_dicom):
    arreglo, imagen_pil = read_dicom_file(archivo_dicom)
    assert arreglo.shape == (64, 64, 3)
    assert arreglo.dtype == np.uint8
    assert imagen_pil.size == (64, 64)


def test_dicom_normaliza_al_rango_visible(archivo_dicom):
    arreglo, _ = read_dicom_file(archivo_dicom)
    assert arreglo.min() >= 0
    assert arreglo.max() <= 255


def test_jpg_devuelve_arreglo_y_pil(archivo_jpg):
    arreglo, imagen_pil = read_jpg_file(archivo_jpg)
    assert arreglo.dtype == np.uint8
    assert arreglo.ndim == 3
    assert imagen_pil is not None


def test_jpg_inexistente_lanza_error():
    with pytest.raises(FileNotFoundError):
        read_jpg_file("no_existe_esta_imagen.jpg")