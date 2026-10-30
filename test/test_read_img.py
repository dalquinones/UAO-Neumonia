"""Pruebas AAA de lectura de imágenes, normalización DICOM y previews."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.models import read_img


def test_read_dicom_devuelve_array_de_tres_canales(ruta_dcm):
    # Arrange — preparar la ruta DICOM real.
    path = ruta_dcm

    # Act — leer el archivo.
    array, _ = read_img.read_dicom(path)

    # Assert — comprobar la forma de la imagen BGR.
    assert array.ndim == 3
    assert array.shape[2] == 3


def test_read_dicom_devuelve_array_uint8(ruta_dcm):
    # Arrange — preparar la ruta DICOM real.
    path = ruta_dcm

    # Act — leer el archivo.
    array, _ = read_img.read_dicom(path)

    # Assert — comprobar el tipo de array y sus elementos.
    assert isinstance(array, np.ndarray)
    assert array.dtype == np.uint8


def test_read_dicom_devuelve_preview_pil_en_grises(ruta_dcm):
    # Arrange — preparar la ruta DICOM real.
    path = ruta_dcm

    # Act — leer el archivo.
    array, preview = read_img.read_dicom(path)

    # Assert — comprobar el contrato de la preview en grises.
    assert isinstance(preview, Image.Image)
    assert preview.mode == "L"
    assert preview.size == (array.shape[1], array.shape[0])


def test_read_dicom_replica_el_gris_en_los_tres_canales(ruta_dcm, monkeypatch):
    # Arrange — simular píxeles distintos sin reemplazar la validación de ruta.
    pixels = np.array([[0, 50, 100], [150, 200, 250]], dtype=np.uint16)
    dicom = SimpleNamespace(pixel_array=pixels)
    monkeypatch.setattr(read_img.pydicom, "dcmread", lambda path: dicom)

    # Act — leer el DICOM simulado.
    array, _ = read_img.read_dicom(ruta_dcm)

    # Assert — comprobar que B, G y R representan el mismo gris.
    np.testing.assert_array_equal(array[:, :, 0], array[:, :, 1])
    np.testing.assert_array_equal(array[:, :, 1], array[:, :, 2])


def test_read_dicom_normaliza_pixeles_de_mas_de_ocho_bits(ruta_dcm, monkeypatch):
    # Arrange — preparar intensidades de 12 bits y su resultado esperado.
    pixels = np.array([[0, 1024], [2048, 4095]], dtype=np.uint16)
    dicom = SimpleNamespace(pixel_array=pixels)
    expected = np.array([[0, 63], [127, 255]], dtype=np.uint8)
    monkeypatch.setattr(read_img.pydicom, "dcmread", lambda path: dicom)

    # Act — normalizar mediante el lector.
    array, _ = read_img.read_dicom(ruta_dcm)

    # Assert — comprobar la escala exacta y el truncamiento a ocho bits.
    np.testing.assert_array_equal(array[:, :, 0], expected)


def test_read_dicom_con_pixeles_cero_devuelve_imagen_negra(ruta_dcm, monkeypatch):
    # Arrange — preparar un DICOM cuyo máximo es cero.
    dicom = SimpleNamespace(pixel_array=np.zeros((2, 3), dtype=np.uint16))
    expected = np.zeros((2, 3, 3), dtype=np.uint8)
    monkeypatch.setattr(read_img.pydicom, "dcmread", lambda path: dicom)

    # Act — leer la imagen sin intensidad.
    array, _ = read_img.read_dicom(ruta_dcm)

    # Assert — comprobar que la salida es completamente negra.
    np.testing.assert_array_equal(array, expected)


def test_read_dicom_recorta_pixeles_negativos_a_cero(ruta_dcm, monkeypatch):
    # Arrange — preparar píxeles con signo y un máximo positivo.
    pixels = np.array([[-100, 0], [100, 200]], dtype=np.int16)
    dicom = SimpleNamespace(pixel_array=pixels)
    expected = np.array([[0, 0], [127, 255]], dtype=np.uint8)
    monkeypatch.setattr(read_img.pydicom, "dcmread", lambda path: dicom)

    # Act — leer el DICOM con intensidades negativas.
    array, _ = read_img.read_dicom(ruta_dcm)

    # Assert — comprobar el recorte sin desbordamiento de negativos.
    np.testing.assert_array_equal(array[:, :, 0], expected)


def test_read_dicom_con_ruta_inexistente_lanza_error(tmp_path):
    # Arrange — preparar una ruta que no existe.
    path = str(tmp_path / "inexistente.dcm")

    # Act — intentar leer el archivo ausente.
    with pytest.raises(FileNotFoundError, match="No se encontró el archivo") as error:
        read_img.read_dicom(path)

    # Assert — comprobar que el error identifica la ruta solicitada.
    assert path in str(error.value)


def test_read_image_devuelve_array_de_tres_canales(ruta_jpg):
    # Arrange — preparar la ruta JPG real.
    path = ruta_jpg

    # Act — leer el archivo.
    array, _ = read_img.read_image(path)

    # Assert — comprobar la forma de la imagen BGR.
    assert array.ndim == 3
    assert array.shape[2] == 3


def test_read_image_devuelve_array_uint8(ruta_jpg):
    # Arrange — preparar la ruta JPG real.
    path = ruta_jpg

    # Act — leer el archivo.
    array, _ = read_img.read_image(path)

    # Assert — comprobar el tipo de array y sus elementos.
    assert isinstance(array, np.ndarray)
    assert array.dtype == np.uint8


def test_read_image_devuelve_preview_pil_rgb(ruta_jpg):
    # Arrange — preparar la ruta JPG real.
    path = ruta_jpg

    # Act — leer el archivo.
    array, preview = read_img.read_image(path)

    # Assert — comprobar el contrato de la preview RGB.
    assert isinstance(preview, Image.Image)
    assert preview.mode == "RGB"
    assert preview.size == (array.shape[1], array.shape[0])


def test_read_image_conserva_los_pixeles_bgr(ruta_jpg, monkeypatch):
    # Arrange — simular la lectura de colores BGR distintos.
    pixels = np.array([[[10, 20, 200], [240, 30, 5]]], dtype=np.uint8)
    expected = pixels.copy()
    monkeypatch.setattr(read_img.cv2, "imread", lambda path: pixels)

    # Act — leer la imagen simulada.
    array, _ = read_img.read_image(ruta_jpg)

    # Assert — comprobar que el array conserva los valores BGR originales.
    np.testing.assert_array_equal(array, expected)


def test_read_image_convierte_preview_de_bgr_a_rgb(ruta_jpg, monkeypatch):
    # Arrange — preparar un píxel cuyos canales azul y rojo son distintos.
    pixels = np.array([[[10, 20, 200]]], dtype=np.uint8)
    monkeypatch.setattr(read_img.cv2, "imread", lambda path: pixels)

    # Act — generar la preview mediante el lector.
    _, preview = read_img.read_image(ruta_jpg)

    # Assert — comprobar el orden RGB del píxel mostrado.
    assert preview.getpixel((0, 0)) == (200, 20, 10)


def test_read_image_con_ruta_inexistente_lanza_error(tmp_path):
    # Arrange — preparar una ruta que no existe.
    path = str(tmp_path / "inexistente.jpg")

    # Act — intentar leer el archivo ausente.
    with pytest.raises(FileNotFoundError, match="No se encontró el archivo") as error:
        read_img.read_image(path)

    # Assert — comprobar que el error identifica la ruta solicitada.
    assert path in str(error.value)


def test_read_image_con_archivo_corrupto_lanza_error(tmp_path):
    # Arrange — crear un archivo existente que no contiene una imagen válida.
    path = tmp_path / "corrupta.jpg"
    path.write_bytes(b"contenido que no representa una imagen")

    # Act — intentar decodificar el archivo corrupto con OpenCV real.
    with pytest.raises(FileNotFoundError, match="No se pudo leer la imagen") as error:
        read_img.read_image(str(path))

    # Assert — comprobar que el error identifica el archivo corrupto.
    assert str(path) in str(error.value)
