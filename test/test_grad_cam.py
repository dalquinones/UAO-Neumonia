"""
Pruebas unitarias para el Módulo 4: grad_cam.py
Valida la inferencia médica y explicabilidad visual (Grad-CAM) sobre radiografías.
"""
import pytest
import numpy as np

from src.grad_cam import predict_and_explain, generate_gradcam_heatmap
from src.load_model import load_cnn_model
from src.preprocess_img import preprocess_image
from src.read_img import read_dicom_file, read_jpg_file

REAL_SAMPLES = [
    ("data/DICOM/normal (2).dcm", "dicom"),
    ("data/DICOM/viral (2).dcm", "dicom"),
    ("data/JPG/normal/NORMAL2-IM-1144-0001.jpeg", "jpg"),
    ("data/JPG/normal/NORMAL2-IM-1145-0001.jpeg", "jpg"),
]


@pytest.fixture(scope="module")
def shared_model():
    """Carga el modelo una sola vez para optimizar tiempo de ejecución."""
    return load_cnn_model()


# 1. Pruebas de inferencia y explicabilidad en muestras reales (4 pruebas)
@pytest.mark.parametrize("filepath,filetype", REAL_SAMPLES, ids=["dcm_norm", "dcm_viral", "jpg_norm1", "jpg_norm2"])
def test_predict_and_explain_real_samples(shared_model, filepath, filetype):
    """Valida inferencia sobre imágenes clínicas reales DICOM y JPG."""
    if filetype == "dicom":
        array, _ = read_dicom_file(filepath)
    else:
        array, _ = read_jpg_file(filepath)

    label, proba, heatmap = predict_and_explain(array, model=shared_model)

    assert label in ["bacteriana", "normal", "viral"]
    assert 0.0 <= proba <= 100.0
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (512, 512, 3)
    assert heatmap.dtype == np.uint8


# 2. Pruebas con entradas sintéticas de diversas dimensiones (3 pruebas)
@pytest.mark.parametrize(
    "shape",
    [(256, 256, 3), (512, 512, 3), (300, 300)],
    ids=["256x256x3", "512x512x3", "300x300_gray"],
)
def test_predict_synthetic_arrays(shared_model, shape):
    """Valida que predict_and_explain acepte arreglos en memoria sin importar su forma."""
    dummy = np.random.randint(50, 200, size=shape, dtype=np.uint8)
    label, proba, heatmap = predict_and_explain(dummy, model=shared_model)

    assert label in ["bacteriana", "normal", "viral"]
    assert heatmap.shape == (512, 512, 3)


# 3. Pruebas de propiedades del heatmap crudo generado (3 pruebas)
@pytest.mark.parametrize(
    "value",
    [0.1, 0.5, 0.9],
    ids=["baja_intensidad", "media_intensidad", "alta_intensidad"],
)
def test_generate_gradcam_heatmap_normalization(shared_model, value):
    """Valida que el heatmap retorne matriz 2D (512x512) normalizada."""
    batch = np.full((1, 512, 512, 1), value, dtype=np.float32)
    heatmap_raw = generate_gradcam_heatmap(shared_model, batch)

    assert heatmap_raw.shape == (512, 512)
    assert heatmap_raw.min() >= 0.0
    assert heatmap_raw.max() <= 1.0


# 4. Pruebas de detección de capas convolucionales inválidas (3 pruebas)
@pytest.mark.parametrize(
    "bad_layer",
    ["capa_fantasma", "dense_1", "conv_no_existe"],
    ids=["fantasma", "dense_1", "conv_err"],
)
def test_predict_invalid_layer_raises_value_error(shared_model, bad_layer):
    """Valida que especificar una capa no convolucional o inexistente lance ValueError."""
    dummy = np.ones((512, 512, 3), dtype=np.uint8) * 100
    with pytest.raises(ValueError):
        predict_and_explain(dummy, model=shared_model, layer_name=bad_layer)


# 5. Pruebas de robustez y casos extremos (2 pruebas)
def test_predict_black_image(shared_model):
    """Valida inferencia en una imagen completamente negra."""
    black_img = np.zeros((512, 512, 3), dtype=np.uint8)
    label, proba, heatmap = predict_and_explain(black_img, model=shared_model)
    assert label in ["bacteriana", "normal", "viral"]
    assert heatmap.shape == (512, 512, 3)


def test_predict_invalid_input_type(shared_model):
    """Valida que pasar un string en vez de ndarray lance TypeError."""
    with pytest.raises(TypeError):
        predict_and_explain("no_es_un_array", model=shared_model)
