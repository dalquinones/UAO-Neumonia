"""
Pruebas unitarias para el Módulo 3: load_model.py
Valida la carga, arquitectura e integridad de la red neuronal convolucional.
"""
import pytest
import tensorflow as tf

from src.load_model import load_cnn_model, validate_model_integrity


@pytest.fixture(scope="module")
def shared_model():
    """Carga el modelo una sola vez para el módulo de pruebas, optimizando tiempo de ejecución."""
    return load_cnn_model()


# 1. Pruebas de propiedades arquitectónicas del modelo (5 pruebas)
def test_load_model_instance(shared_model):
    """Valida que sea una instancia válida de tf.keras.Model."""
    assert isinstance(shared_model, tf.keras.Model)


def test_model_input_dimensions(shared_model):
    """Valida que la forma de entrada sea (None, 512, 512, 1)."""
    input_shape = shared_model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    assert input_shape[1:] == (512, 512, 1)


def test_model_output_dimensions(shared_model):
    """Valida que la capa final tenga 3 salidas (normal, bacteria, virus)."""
    output_shape = shared_model.output_shape
    if isinstance(output_shape, list):
        output_shape = output_shape[0]
    assert output_shape[-1] == 3


def test_model_has_required_layers_count(shared_model):
    """Valida que el modelo contenga la profundidad de capas esperada."""
    assert len(shared_model.layers) >= 10, "El modelo debe tener al menos 10 capas"


def test_model_has_conv10_layer(shared_model):
    """Valida la presencia específica de la capa conv10_thisone para Grad-CAM."""
    layer = shared_model.get_layer("conv10_thisone")
    assert layer is not None
    assert "conv" in layer.name.lower()


# 2. Pruebas de validación de capas inexistentes (4 pruebas)
@pytest.mark.parametrize(
    "missing_layer",
    [
        "capa_inexistente_1",
        "conv_falsa_99",
        "dense_inexistente",
        "lstm_layer_error",
    ],
    ids=["falsa_1", "conv_99", "dense_err", "lstm_err"],
)
def test_validate_integrity_detects_missing_layers(shared_model, missing_layer):
    """Valida que validate_model_integrity lance ValueError ante capas ausentes."""
    with pytest.raises(ValueError, match="No se encontró la capa"):
        validate_model_integrity(shared_model, required_layer=missing_layer)


# 3. Pruebas de rutas de archivo inexistentes (4 pruebas)
@pytest.mark.parametrize(
    "missing_path",
    [
        "modelo_fantasma.h5",
        "pesos/no_existe.h5",
        "red_neuronal_falsa.keras",
        "carpeta_vacia/modelo.h5",
    ],
    ids=["fantasma", "no_existe", "falsa_keras", "carpeta_vacia"],
)
def test_load_model_missing_files(missing_path):
    """Valida que lance FileNotFoundError al buscar archivos h5 inexistentes."""
    with pytest.raises(FileNotFoundError):
        load_cnn_model(missing_path)


# 4. Pruebas de paso de objetos inválidos a validate_model_integrity (2 pruebas)
@pytest.mark.parametrize(
    "invalid_obj",
    ["no_es_un_modelo", None],
    ids=["string", "None"],
)
def test_validate_integrity_invalid_model_object(invalid_obj):
    """Valida que objetos que no sean modelos disparen AttributeError o TypeError."""
    with pytest.raises((AttributeError, TypeError)):
        validate_model_integrity(invalid_obj)
