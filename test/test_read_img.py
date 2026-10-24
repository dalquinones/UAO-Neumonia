"""
Pruebas unitarias para el Módulo 1: read_img.py
Valida la lectura de imágenes radiográficas DICOM (.dcm) y estándar (.jpg, .jpeg, .png).
"""
import os
import pytest
import numpy as np
from PIL import Image

from src.read_img import read_dicom_file, read_jpg_file, read_file

# Muestras reales en el repositorio
REAL_DICOM_SAMPLES = [
    "data/DICOM/normal (2).dcm",
    "data/DICOM/normal (3).dcm",
    "data/DICOM/viral (2).dcm",
    "data/DICOM/viral (3).dcm",
]

REAL_JPG_SAMPLES = [
    "data/JPG/normal/NORMAL2-IM-1144-0001.jpeg",
    "data/JPG/normal/NORMAL2-IM-1145-0001.jpeg",
]


# 1. Pruebas de lectura DICOM reales (4 pruebas)
@pytest.mark.parametrize("dcm_path", REAL_DICOM_SAMPLES)
def test_read_dicom_real_samples(dcm_path):
    """Valida lectura de archivos clínicos DICOM existentes."""
    array, img_pil = read_dicom_file(dcm_path)

    assert isinstance(array, np.ndarray), "Debe retornar un arreglo NumPy"
    assert isinstance(img_pil, Image.Image), "Debe retornar un objeto PIL Image"
    assert array.dtype == np.uint8, "El tipo debe ser uint8"
    assert len(array.shape) == 3 and array.shape[2] == 3, "Debe tener 3 canales RGB"
    assert array.min() >= 0 and array.max() <= 255, "Rango 0-255"


# 2. Pruebas de lectura JPG/JPEG reales (2 pruebas)
@pytest.mark.parametrize("jpg_path", REAL_JPG_SAMPLES)
def test_read_jpg_real_samples(jpg_path):
    """Valida lectura de imágenes JPG existentes."""
    array, img_pil = read_jpg_file(jpg_path)

    assert isinstance(array, np.ndarray)
    assert isinstance(img_pil, Image.Image)
    assert array.dtype == np.uint8
    assert len(array.shape) == 3 and array.shape[2] == 3


# 3. Pruebas de la función fachada read_file con detección automática (6 pruebas)
@pytest.mark.parametrize("sample_path", REAL_DICOM_SAMPLES + REAL_JPG_SAMPLES)
def test_read_file_auto_detect_all_real_samples(sample_path):
    """Valida que read_file reconozca tanto DICOM como JPG reales."""
    array, img_pil = read_file(sample_path)
    assert isinstance(array, np.ndarray)
    assert isinstance(img_pil, Image.Image)


# 4. Pruebas de extensiones generadas dinámicamente (7 pruebas)
@pytest.mark.parametrize(
    "extension",
    [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"],
)
def test_read_file_extension_case_insensitivity(tmp_path, extension):
    """Valida que soporte mayúsculas y minúsculas en extensiones de imagen estándar."""
    test_file = tmp_path / f"test_sample{extension}"
    synthetic_pil = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 120)
    synthetic_pil.save(str(test_file), format="PNG")

    array, img_pil = read_file(str(test_file))
    assert isinstance(array, np.ndarray)
    assert array.shape == (100, 100, 3)


def test_read_file_dcm_uppercase_extension(tmp_path):
    """Valida que soporte la extensión .DCM en mayúsculas para archivos DICOM."""
    import shutil
    dcm_copy = tmp_path / "SAMPLE_UPPER.DCM"
    shutil.copy("data/DICOM/normal (2).dcm", str(dcm_copy))

    array, img_pil = read_file(str(dcm_copy))
    assert isinstance(array, np.ndarray)
    assert array.shape[2] == 3


# 5. Pruebas de rutas inexistentes (FileNotFoundError) (4 pruebas)
@pytest.mark.parametrize(
    "missing_path",
    [
        "no_existe/radiografia.dcm",
        "data/falso/archivo.jpg",
        "sin_carpeta/test.jpeg",
        "ruta_inventada/sin_extension",
    ],
)
def test_read_file_missing_paths(missing_path):
    """Valida que dispare FileNotFoundError si la ruta no existe."""
    with pytest.raises(FileNotFoundError):
        read_file(missing_path)


# 6. Pruebas de archivos vacíos o directorios (2 pruebas)
def test_read_jpg_corrupt_or_empty(tmp_path):
    """Valida que un archivo vacío de 0 bytes dispare ValueError en read_jpg."""
    empty_file = tmp_path / "empty.jpg"
    empty_file.write_bytes(b"")
    with pytest.raises(ValueError):
        read_jpg_file(str(empty_file))


def test_read_file_directory_path(tmp_path):
    """Valida que pasar un directorio en vez de archivo falle adecuadamente."""
    with pytest.raises((ValueError, FileNotFoundError, IsADirectoryError, Exception)):
        read_jpg_file(str(tmp_path))
