"""
Pruebas unitarias para el Módulo 5: integrator.py
Valida la orquestación integral del pipeline de diagnóstico, caché y puente de interfaz.
"""
import pytest
import numpy as np
from PIL import Image

from src.integrator import process_and_diagnose, predict
from src.load_model import load_cnn_model

REAL_SAMPLES = [
    "data/DICOM/normal (2).dcm",
    "data/DICOM/normal (3).dcm",
    "data/DICOM/viral (2).dcm",
    "data/DICOM/viral (3).dcm",
    "data/JPG/normal/NORMAL2-IM-1144-0001.jpeg",
    "data/JPG/normal/NORMAL2-IM-1145-0001.jpeg",
]


# 1. Pruebas de orquestación completa con rutas de archivos reales (6 pruebas)
@pytest.mark.parametrize("sample_path", REAL_SAMPLES)
def test_process_and_diagnose_real_files(sample_path):
    """Valida la ejecución del pipeline completo de principio a fin."""
    results = process_and_diagnose(sample_path)

    assert isinstance(results, dict)
    assert results["label"] in ["bacteriana", "normal", "viral"]
    assert 0.0 <= results["probability"] <= 100.0
    assert isinstance(results["heatmap"], np.ndarray)
    assert results["heatmap"].shape == (512, 512, 3)
    assert isinstance(results["display_image"], Image.Image)


# 2. Pruebas de orquestación con arreglos NumPy en memoria (3 pruebas)
@pytest.mark.parametrize(
    "shape",
    [(128, 128, 3), (256, 256, 3), (512, 512, 3)],
    ids=["128x128", "256x256", "512x512"],
)
def test_process_and_diagnose_memory_arrays(shape):
    """Valida orquestación cuando se entrega directamente un array en RAM."""
    dummy_arr = np.random.randint(20, 220, size=shape, dtype=np.uint8)
    results = process_and_diagnose(dummy_arr)

    assert results["label"] in ["bacteriana", "normal", "viral"]
    assert isinstance(results["heatmap"], np.ndarray)


# 3. Pruebas de la función puente predict() para la GUI (3 pruebas)
@pytest.mark.parametrize(
    "shape",
    [(200, 200, 3), (512, 512, 3), (400, 400)],
    ids=["200x200", "512x512", "400x400_gray"],
)
def test_predict_bridge_function(shape):
    """Valida la compatibilidad del puente predict(array) con la interfaz gráfica."""
    dummy = np.ones(shape, dtype=np.uint8) * 128
    label, proba, heatmap = predict(dummy)

    assert label in ["bacteriana", "normal", "viral"]
    assert isinstance(proba, float)
    assert isinstance(heatmap, np.ndarray)


# 4. Prueba de reutilización y caché del modelo (1 prueba)
def test_model_caching_singleton():
    """Valida que load_cnn_model no recargue el archivo del disco repetidamente."""
    model_a = load_cnn_model()
    model_b = load_cnn_model()
    assert model_a is model_b, "Debe retornar la misma referencia de modelo"


# 5. Pruebas de tipos de entrada inválidos (4 pruebas)
@pytest.mark.parametrize(
    "invalid_data",
    [None, 9999, [1, 2, 3], {"clave": "valor"}],
    ids=["None", "int", "list", "dict"],
)
def test_process_and_diagnose_invalid_types(invalid_data):
    """Valida que tipos no soportados disparen TypeError."""
    with pytest.raises(TypeError):
        process_and_diagnose(invalid_data)


# 6. Pruebas de rutas de archivo inexistentes (3 pruebas)
@pytest.mark.parametrize(
    "missing_path",
    ["no_existe/radiografia.dcm", "falsa/imagen.jpg", "error/foto.png"],
    ids=["dcm_err", "jpg_err", "png_err"],
)
def test_process_and_diagnose_missing_files(missing_path):
    """Valida que rutas que no existen disparen FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        process_and_diagnose(missing_path)
