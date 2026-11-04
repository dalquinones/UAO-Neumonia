"""Pruebas sobre la explicación visual sin cargar un modelo neuronal."""

import numpy as np
import pytest
from PIL import Image

from src.neumonia.grad_cam import superponer_mapa


def test_superposicion_conserva_dimensiones():
    original = Image.new("RGB", (30, 20), color=(100, 100, 100))
    saliencia = np.ones((20, 30), dtype=np.float32) * 0.75
    resultado = superponer_mapa(original, saliencia)
    assert resultado.size == original.size
    assert resultado.mode == "RGB"


def test_superposicion_rechaza_dimensiones_incompatibles():
    with pytest.raises(ValueError, match="tamaño"):
        superponer_mapa(Image.new("RGB", (10, 10)), np.zeros((9, 10)))
