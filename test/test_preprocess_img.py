"""Treinta pruebas AAA del contrato y las etapas de preprocesamiento.

Discrepancia: test/README.md menciona que se aceptan dos canales. La implementación
acepta gris 2D o un canal, pero rechaza dos canales; aquí se prueba el código real.
Los tipos no admitidos provocan errores de OpenCV, no una validación propia.
"""

from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from src.models import preprocess_img


@pytest.fixture
def imagen_bgr():
    """Patrón reproducible con canales distintos e intensidades variadas."""
    return (np.arange(37 * 59 * 3).reshape(37, 59, 3) % 256).astype(np.uint8)


def test_preprocess_entrega_forma_de_inferencia():
    # Arrange
    imagen = np.full((512, 512, 3), 100, dtype=np.uint8)
    # Act
    resultado = preprocess_img.preprocess(imagen)
    # Assert
    assert resultado.shape == (1, 512, 512, 1)


def test_preprocess_amplia_imagen_pequena():
    # Arrange
    imagen = np.full((16, 16, 3), 100, dtype=np.uint8)
    # Act
    resultado = preprocess_img.preprocess(imagen)
    # Assert
    assert resultado.shape == (1, 512, 512, 1)


def test_preprocess_reduce_imagen_grande():
    # Arrange
    imagen = np.full((768, 1024, 3), 100, dtype=np.uint8)
    # Act
    resultado = preprocess_img.preprocess(imagen)
    # Assert
    assert resultado.shape == (1, 512, 512, 1)


def test_preprocess_adapta_imagen_horizontal():
    # Arrange
    imagen = np.full((32, 160, 3), 100, dtype=np.uint8)
    # Act
    resultado = preprocess_img.preprocess(imagen)
    # Assert
    assert resultado.shape == (1, 512, 512, 1)


def test_preprocess_adapta_imagen_vertical():
    # Arrange
    imagen = np.full((160, 32, 3), 100, dtype=np.uint8)
    # Act
    resultado = preprocess_img.preprocess(imagen)
    # Assert
    assert resultado.shape == (1, 512, 512, 1)


def test_preprocess_adapta_imagen_de_un_pixel():
    # Arrange
    imagen = np.full((1, 1, 3), 100, dtype=np.uint8)
    # Act
    resultado = preprocess_img.preprocess(imagen)
    # Assert
    assert resultado.shape == (1, 512, 512, 1)


def test_preprocess_adapta_dimensiones_impares(imagen_bgr):
    # Arrange
    imagen = imagen_bgr
    # Act
    resultado = preprocess_img.preprocess(imagen)
    # Assert
    assert resultado.shape == (1, 512, 512, 1)


def test_preprocess_acepta_gris_bidimensional():
    # Arrange
    imagen = np.full((37, 59), 100, dtype=np.uint8)
    # Act
    resultado = preprocess_img.preprocess(imagen)
    # Assert
    assert resultado.shape == (1, 512, 512, 1)


def test_preprocess_acepta_gris_con_eje_de_canal():
    # Arrange
    imagen = np.full((37, 59, 1), 100, dtype=np.uint8)
    # Act
    resultado = preprocess_img.preprocess(imagen)
    # Assert
    assert resultado.shape == (1, 512, 512, 1)


def test_preprocess_gris_equivale_a_bgr_replicado():
    # Arrange
    gris = (np.arange(37 * 59).reshape(37, 59) % 256).astype(np.uint8)
    bgr = np.repeat(gris[:, :, np.newaxis], 3, axis=2)
    esperado = preprocess_img.preprocess(bgr)
    # Act
    resultado = preprocess_img.preprocess(gris)
    # Assert
    np.testing.assert_array_equal(resultado, esperado)


def test_preprocess_devuelve_float32(imagen_bgr):
    # Arrange
    imagen = imagen_bgr
    # Act
    resultado = preprocess_img.preprocess(imagen)
    # Assert
    assert resultado.dtype == np.dtype(np.float32)


def test_preprocess_mantiene_rango_normalizado(imagen_bgr):
    # Arrange
    imagen = imagen_bgr
    # Act
    resultado = preprocess_img.preprocess(imagen)
    # Assert
    assert np.all((resultado >= 0.0) & (resultado <= 1.0))


def test_preprocess_devuelve_valores_finitos(imagen_bgr):
    # Arrange
    imagen = imagen_bgr
    # Act
    resultado = preprocess_img.preprocess(imagen)
    # Assert
    assert np.isfinite(resultado).all()


def test_preprocess_imagen_plana_conserva_uniformidad():
    # Arrange
    imagen = np.full((37, 59, 3), 100, dtype=np.uint8)
    # Act
    resultado = preprocess_img.preprocess(imagen)
    # Assert
    assert np.all(resultado == resultado[0, 0, 0, 0])


def test_preprocess_es_determinista(imagen_bgr):
    # Arrange
    esperado = preprocess_img.preprocess(imagen_bgr)
    # Act
    resultado = preprocess_img.preprocess(imagen_bgr)
    # Assert
    np.testing.assert_array_equal(resultado, esperado)


def test_preprocess_no_modifica_entrada(imagen_bgr):
    # Arrange
    copia = imagen_bgr.copy()
    # Act
    preprocess_img.preprocess(imagen_bgr)
    # Assert
    np.testing.assert_array_equal(imagen_bgr, copia)


def test_preprocess_usa_tamano_objetivo_de_entrenamiento(imagen_bgr, monkeypatch):
    # Arrange
    resize = Mock(wraps=cv2.resize)
    monkeypatch.setattr(preprocess_img.cv2, "resize", resize)
    # Act
    preprocess_img.preprocess(imagen_bgr)
    # Assert
    resize.assert_called_once_with(imagen_bgr, (512, 512))


def test_preprocess_configura_clahe_como_entrenamiento(imagen_bgr, monkeypatch):
    # Arrange
    crear_clahe = Mock(wraps=cv2.createCLAHE)
    monkeypatch.setattr(preprocess_img.cv2, "createCLAHE", crear_clahe)
    # Act
    preprocess_img.preprocess(imagen_bgr)
    # Assert
    crear_clahe.assert_called_once_with(clipLimit=2.0, tileGridSize=(4, 4))


def test_preprocess_interpreta_canales_como_bgr(monkeypatch):
    # Arrange
    imagen = np.zeros((512, 512, 3), dtype=np.uint8)
    imagen[:, :256] = (255, 0, 0)
    imagen[:, 256:] = (0, 0, 255)
    esperado = np.empty((1, 512, 512, 1), dtype=np.float32)
    # Pesos BGR: azul puro produce 29 y rojo puro 76 en gris uint8.
    esperado[:, :, :256, :] = 29 / 255.0
    esperado[:, :, 256:, :] = 76 / 255.0
    clahe = Mock()
    clahe.apply.side_effect = lambda gris: gris
    monkeypatch.setattr(preprocess_img.cv2, "createCLAHE", Mock(return_value=clahe))
    # Act
    resultado = preprocess_img.preprocess(imagen)
    # Assert
    np.testing.assert_allclose(resultado, esperado, rtol=1e-6, atol=0)


def test_preprocess_normaliza_por_255_sin_minmax(imagen_bgr, monkeypatch):
    # Arrange
    # Sin extremos 0/255, una normalización min-max daría un resultado distinto.
    realzada = np.resize(np.array([64, 128, 192], dtype=np.uint8), (512, 512))
    esperado = np.resize(np.array([64 / 255, 128 / 255, 192 / 255]), (1, 512, 512, 1))
    clahe = Mock()
    clahe.apply.return_value = realzada
    monkeypatch.setattr(preprocess_img.cv2, "createCLAHE", Mock(return_value=clahe))
    # Act
    resultado = preprocess_img.preprocess(imagen_bgr)
    # Assert
    np.testing.assert_allclose(resultado, esperado, rtol=1e-6, atol=0)


def test_preprocess_rechaza_none():
    # Arrange
    imagen = None
    # Act
    with pytest.raises(ValueError) as error:
        preprocess_img.preprocess(imagen)
    # Assert
    assert "vacío" in str(error.value)


def test_preprocess_rechaza_array_vacio():
    # Arrange
    imagen = np.array([], dtype=np.uint8)
    # Act
    with pytest.raises(ValueError) as error:
        preprocess_img.preprocess(imagen)
    # Assert
    assert "vacío" in str(error.value)


def test_preprocess_rechaza_altura_cero():
    # Arrange
    imagen = np.empty((0, 10, 3), dtype=np.uint8)
    # Act
    with pytest.raises(ValueError) as error:
        preprocess_img.preprocess(imagen)
    # Assert
    assert "vacío" in str(error.value)


def test_preprocess_rechaza_anchura_cero():
    # Arrange
    imagen = np.empty((10, 0, 3), dtype=np.uint8)
    # Act
    with pytest.raises(ValueError) as error:
        preprocess_img.preprocess(imagen)
    # Assert
    assert "vacío" in str(error.value)


def test_preprocess_rechaza_dos_canales():
    # Arrange — dos canales no equivalen a una imagen gris 2D.
    imagen = np.zeros((37, 59, 2), dtype=np.uint8)
    # Act
    with pytest.raises(ValueError) as error:
        preprocess_img.preprocess(imagen)
    # Assert
    assert "recibí 2" in str(error.value)


def test_preprocess_rechaza_cuatro_canales():
    # Arrange
    imagen = np.zeros((37, 59, 4), dtype=np.uint8)
    # Act
    with pytest.raises(ValueError) as error:
        preprocess_img.preprocess(imagen)
    # Assert
    assert "recibí 4" in str(error.value)


def test_preprocess_rechaza_float32_en_clahe():
    # Arrange
    imagen = np.zeros((37, 59, 3), dtype=np.float32)
    # Act
    with pytest.raises(cv2.error) as error:
        preprocess_img.preprocess(imagen)
    # Assert
    assert "clahe.cpp" in str(error.value)


def test_preprocess_rechaza_float64():
    # Arrange
    imagen = np.zeros((37, 59, 3), dtype=np.float64)
    # Act
    with pytest.raises(cv2.error) as error:
        preprocess_img.preprocess(imagen)
    # Assert
    assert "Unsupported depth" in str(error.value)


def test_preprocess_rechaza_int32():
    # Arrange
    imagen = np.zeros((37, 59, 3), dtype=np.int32)
    # Act
    with pytest.raises(cv2.error) as error:
        preprocess_img.preprocess(imagen)
    # Assert
    assert "resize" in str(error.value)


def test_preprocess_rechaza_booleanos():
    # Arrange
    imagen = np.zeros((37, 59, 3), dtype=np.bool_)
    # Act
    with pytest.raises(cv2.error) as error:
        preprocess_img.preprocess(imagen)
    # Assert
    assert "not supported" in str(error.value)
