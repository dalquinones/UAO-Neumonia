"""Pruebas del modulo Grad-CAM."""

import numpy as np
import pytest

from src.grad_cam import CAPA_CONVOLUCIONAL_FINAL, TAMANO_SALIDA, grad_cam
from src.load_model import load_model
from src.preprocess_img import preprocess

from .conftest import hay_modelo


@hay_modelo
def test_forma_de_salida_es_correcta(imagen_rgb):
    """La imagen de salida debe tener forma (512, 512, 3)."""
    resultado = grad_cam(imagen_rgb)
    assert resultado.shape == (TAMANO_SALIDA[1], TAMANO_SALIDA[0], 3)


@hay_modelo
def test_tipo_de_salida_es_uint8(imagen_rgb):
    """La imagen de salida debe ser uint8."""
    resultado = grad_cam(imagen_rgb)
    assert resultado.dtype == np.uint8


@hay_modelo
def test_mapa_de_activacion_normalizado_entre_cero_y_uno(imagen_rgb):
    """El mapa de activacion debe estar normalizado entre 0 y 1."""
    modelo = load_model()
    lote = preprocess(imagen_rgb)
    mapa = _calcular_mapa(modelo, lote, CAPA_CONVOLUCIONAL_FINAL)
    assert mapa.min() >= 0.0
    assert mapa.max() <= 1.0


@hay_modelo
def test_capa_inexistente_lanza_error_claro(imagen_rgb):
    """Debe lanzar ValueError con mensaje claro si la capa no existe."""
    modelo = load_model()
    lote = preprocess(imagen_rgb)
    with pytest.raises(ValueError, match="No such layer"):
        _calcular_mapa(modelo, lote, "capa_que_no_existe")


@hay_modelo
def test_grad_cam_con_capa_inexistente_lanza_error(imagen_rgb):
    """grad_cam debe propagar el error si la capa no existe."""
    with pytest.raises(ValueError):
        grad_cam(imagen_rgb, nombre_capa="capa_que_no_existe")


@hay_modelo
def test_grad_cam_con_modelo_proporcionado(imagen_rgb):
    """Debe funcionar cuando se pasa un modelo ya cargado."""
    modelo = load_model()
    resultado = grad_cam(imagen_rgb, modelo=modelo)
    assert resultado.shape == (TAMANO_SALIDA[1], TAMANO_SALIDA[0], 3)
    assert resultado.dtype == np.uint8


@hay_modelo
def test_grad_cam_es_determinista(imagen_rgb):
    """La salida debe ser determinista para la misma entrada."""
    resultado1 = grad_cam(imagen_rgb)
    resultado2 = grad_cam(imagen_rgb)
    np.testing.assert_array_equal(resultado1, resultado2)
