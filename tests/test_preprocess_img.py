"""Pruebas del modulo de preprocesamiento."""

import numpy as np

from src.preprocess_img import TAMANO_ENTRADA, preprocess


def test_forma_de_salida_es_lote_de_un_canal(imagen_rgb):
    resultado = preprocess(imagen_rgb)
    assert resultado.shape == (1, TAMANO_ENTRADA[1], TAMANO_ENTRADA[0], 1)


def test_valores_normalizados_entre_cero_y_uno(imagen_rgb):
    resultado = preprocess(imagen_rgb)
    assert resultado.min() >= 0.0
    assert resultado.max() <= 1.0


def test_salida_independiente_del_tamano_de_entrada():
    pequena = np.full((50, 80, 3), 120, dtype=np.uint8)
    grande = np.full((900, 1200, 3), 120, dtype=np.uint8)
    assert preprocess(pequena).shape == preprocess(grande).shape


def test_es_determinista(imagen_rgb):
    np.testing.assert_array_equal(preprocess(imagen_rgb), preprocess(imagen_rgb))