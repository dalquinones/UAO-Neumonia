"""
Pruebas unitarias para el Módulo 2: preprocess_img.py
Valida la preparación matemática y morfológica de imágenes para la CNN.
"""
import pytest
import numpy as np

from src.preprocess_img import preprocess_image
from src.read_img import read_dicom_file

SAMPLE_DICOM = "data/DICOM/normal (2).dcm"


# 1. Pruebas de dimensiones y resolución de entrada (7 pruebas)
@pytest.mark.parametrize(
    "input_shape",
    [
        (100, 100),
        (256, 256),
        (512, 512),
        (1024, 768),
        (333, 517),
        (600, 600, 3),
        (256, 256, 1),
    ],
    ids=[
        "pequeña-100x100",
        "mediana-256x256",
        "estándar-512x512",
        "rectangular-1024x768",
        "impar-333x517",
        "3-canales-600x600x3",
        "1-canal-256x256x1",
    ],
)
def test_preprocess_various_input_shapes(input_shape):
    """Valida que cualquier resolución o canal se redimensione a (1, 512, 512, 1)."""
    dummy_img = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
    tensor = preprocess_image(dummy_img)

    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 512, 512, 1)


# 2. Pruebas de tipos de datos de NumPy (5 pruebas)
@pytest.mark.parametrize(
    "dtype",
    [np.uint8, np.uint16, np.int32, np.float32, np.float64],
    ids=["uint8", "uint16", "int32", "float32", "float64"],
)
def test_preprocess_supported_dtypes(dtype):
    """Valida que soporte diversos formatos numéricos y siempre retorne float32."""
    raw_array = np.random.randint(0, 100, size=(256, 256)).astype(dtype)
    tensor = preprocess_image(raw_array)

    assert tensor.dtype == np.float32
    assert tensor.shape == (1, 512, 512, 1)


# 3. Pruebas de normalización y rangos extremos de intensidad (6 pruebas)
@pytest.mark.parametrize(
    "fill_value,description",
    [
        (0, "imagen-completamente-negra"),
        (255, "imagen-completamente-blanca"),
        (128, "imagen-gris-medio"),
        (0.5, "rango-flotante-0-a-1"),
        (1000, "escala-hounsfield-alta"),
        (4095, "dicom-12bits"),
    ],
    ids=["negra", "blanca", "gris", "float_unitario", "hounsfield", "12bits"],
)
def test_preprocess_pixel_ranges(fill_value, description):
    """Valida que sin importar la escala de entrada, la salida esté entre [0.0, 1.0]."""
    array = np.zeros((300, 300), dtype=np.float32)
    if fill_value > 0:
        array[150:, :] = fill_value
    tensor = preprocess_image(array)

    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_constant_image_equivalence():
    """Valida que imágenes constantes equivalentes en distintos dtypes produzcan el mismo tensor."""
    img_uint8 = preprocess_image(np.full((256, 256), 128, dtype=np.uint8))
    img_uint16 = preprocess_image(np.full((256, 256), 32896, dtype=np.uint16))
    img_float = preprocess_image(np.full((256, 256), 0.5, dtype=np.float32))

    # Verificar que las 3 representaciones de "gris medio" generen tensores equivalentes tras CLAHE
    np.testing.assert_allclose(img_uint8, img_float, atol=1e-2)
    np.testing.assert_allclose(img_uint16, img_float, atol=1e-2)


# 4. Pruebas de validación de tipos erróneos (5 pruebas)
@pytest.mark.parametrize(
    "invalid_input",
    [
        "ruta/a/imagen.jpg",
        12345,
        [1, 2, 3],
        {"imagen": np.zeros((10, 10))},
        None,
    ],
    ids=["string", "entero", "lista", "diccionario", "None"],
)
def test_preprocess_invalid_types_raise_type_error(invalid_input):
    """Valida que tipos que no sean ndarray disparen TypeError."""
    with pytest.raises(TypeError):
        preprocess_image(invalid_input)


# 5. Pruebas de validación de arreglos corruptos o no soportados (3 pruebas)
@pytest.mark.parametrize(
    "invalid_array",
    [
        np.array([]),                    # Vacío
        np.zeros((10,)),                 # 1D
        np.zeros((512, 512, 5)),         # 5 canales no soportados
    ],
    ids=["arreglo_vacio", "arreglo_1D", "cinco_canales"],
)
def test_preprocess_invalid_arrays_raise_value_error(invalid_array):
    """Valida que arreglos vacíos o con dimensiones incompatibles disparen ValueError."""
    with pytest.raises(ValueError):
        preprocess_image(invalid_array)
