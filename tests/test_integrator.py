"""Pruebas del orquestador del pipeline."""

import numpy as np

from src.integrator import ETIQUETAS, predict

from .conftest import hay_modelo


def test_las_etiquetas_cubren_las_tres_clases():
    assert set(ETIQUETAS.values()) == {"bacteriana", "normal", "viral"}


@hay_modelo
def test_prediccion_devuelve_etiqueta_probabilidad_y_heatmap(imagen_rgb):
    etiqueta, probabilidad, mapa = predict(imagen_rgb)
    assert etiqueta in ETIQUETAS.values()
    assert 0.0 <= probabilidad <= 100.0
    assert mapa.shape == (512, 512, 3)
    assert mapa.dtype == np.uint8