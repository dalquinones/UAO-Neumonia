"""Pruebas del modulo de carga del modelo."""

import pytest

from src.load_model import load_model

from .conftest import hay_modelo


def test_ruta_inexistente_lanza_error():
    with pytest.raises(FileNotFoundError):
        load_model("modelo_que_no_existe.h5")


@hay_modelo
def test_el_modelo_se_carga_una_sola_vez():
    """La cache debe devolver exactamente el mismo objeto."""
    assert load_model() is load_model()


@hay_modelo
def test_el_modelo_tiene_la_capa_de_grad_cam():
    from src.grad_cam import CAPA_CONVOLUCIONAL_FINAL

    nombres = [capa.name for capa in load_model().layers]
    assert CAPA_CONVOLUCIONAL_FINAL in nombres