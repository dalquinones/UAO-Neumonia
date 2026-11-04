"""Pruebas para el contrato del preprocesamiento explicado en clase 1."""

import numpy as np
import pytest
from PIL import Image

from src.neumonia.preprocess_img import preprocess_image


def test_preprocesamiento_shape_rango_tipo():
    # Arrange
    imagen = Image.fromarray(np.full((100, 200, 3), 120, dtype=np.uint8))
    # Act
    tensor = preprocess_image(imagen)
    # Assert
    assert tensor.shape == (1, 512, 512, 1)
    assert tensor.dtype == np.float32
    assert 0 <= tensor.min() <= tensor.max() <= 1


def test_preprocesamiento_escala_grises():
    imagen = Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8))
    salida = preprocess_image(imagen)
    assert salida.shape == (1, 512, 512, 1)
    assert np.all(np.isfinite(salida))


def test_preprocesamiento_imagen_constante():
    salida = preprocess_image(Image.new("RGB", (20, 20), color=(0, 0, 0)))
    assert np.all(np.isfinite(salida))
    assert salida.min() >= 0
    assert salida.max() <= 1


@pytest.mark.parametrize("malformado", [np.array([]), np.zeros((2, 2, 2, 2))])
def test_rechaza_datos_sin_forma_de_imagen(malformado):
    with pytest.raises(ValueError, match="imagen|Imagen"):
        preprocess_image(malformado)
