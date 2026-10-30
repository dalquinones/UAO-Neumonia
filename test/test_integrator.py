"""Pruebas AAA de la orquestación de Integrator, sin TensorFlow.

El controlador delega la lectura en read_img.read_file (el despacho por extensión
se prueba en test_read_img.py) y conserva la preview del lector, incluso si es gris
(como ocurre con DICOM).
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from src.controllers import integrator as modulo_integrator
from src.models.preprocess_img import preprocess


@pytest.fixture
def pipeline(integrator, monkeypatch):
    """Aísla lectura y preprocesamiento para probar solo la orquestación."""
    array = np.full((8, 8, 3), 100, dtype=np.uint8)
    preview = Image.new("RGB", (8, 8), color="red")
    tensor = np.full((1, 512, 512, 1), 0.5, dtype=np.float32)
    lectura = Mock(return_value=(array, preview))
    preprocesar = Mock(return_value=tensor)
    monkeypatch.setattr(integrator, "_read_file", lectura)
    monkeypatch.setattr(modulo_integrator, "preprocess", preprocesar)
    return SimpleNamespace(
        array=array, preview=preview, tensor=tensor, lectura=lectura, preprocesar=preprocesar
    )


def test_analyze_predice_bacteriana(integrator, pipeline, fakemodel_factory):
    # Arrange
    integrator.model = fakemodel_factory(clase_programada=0)
    # Act
    resultado = integrator.analyze("entrada.jpg")
    # Assert
    assert resultado["label"] == "bacteriana"


def test_analyze_predice_normal(integrator, pipeline, fakemodel_factory):
    # Arrange
    integrator.model = fakemodel_factory(clase_programada=1)
    # Act
    resultado = integrator.analyze("entrada.jpg")
    # Assert
    assert resultado["label"] == "normal"


def test_analyze_predice_viral(integrator, pipeline, fakemodel_factory):
    # Arrange
    integrator.model = fakemodel_factory(clase_programada=2)
    # Act
    resultado = integrator.analyze("entrada.jpg")
    # Assert
    assert resultado["label"] == "viral"


def test_analyze_devuelve_diccionario_con_claves_exactas(integrator, pipeline):
    # Arrange
    claves = {"label", "probabilities", "preview"}
    # Act
    resultado = integrator.analyze("entrada.jpg")
    # Assert
    assert isinstance(resultado, dict) and set(resultado) == claves


def test_analyze_devuelve_probabilidades_como_lista(integrator, pipeline):
    # Arrange
    ruta = "entrada.jpg"
    # Act
    resultado = integrator.analyze(ruta)
    # Assert
    assert isinstance(resultado["probabilities"], list)


def test_analyze_devuelve_tres_probabilidades(integrator, pipeline):
    # Arrange
    ruta = "entrada.jpg"
    # Act
    resultado = integrator.analyze(ruta)
    # Assert
    assert len(resultado["probabilities"]) == 3


def test_analyze_convierte_probabilidades_a_float_nativo(integrator, pipeline):
    # Arrange
    ruta = "entrada.jpg"
    # Act
    resultado = integrator.analyze(ruta)
    # Assert
    assert all(type(valor) is float for valor in resultado["probabilities"])


def test_analyze_preserva_valores_y_orden_de_probabilidades(integrator, pipeline, fake_model):
    # Arrange
    fake_model.clase_programada = 1
    # Act
    resultado = integrator.analyze("entrada.jpg")
    # Assert
    assert resultado["probabilities"] == pytest.approx([0.05, 0.90, 0.05])


def test_analyze_incluye_previsualizacion_pil(integrator, pipeline):
    # Arrange
    ruta = "entrada.jpg"
    # Act
    resultado = integrator.analyze(ruta)
    # Assert
    assert isinstance(resultado["preview"], Image.Image)


def test_analyze_conserva_identidad_de_previsualizacion(integrator, pipeline):
    # Arrange — también se conserva una preview gris, válida para DICOM.
    preview = Image.new("L", (8, 8), color=100)
    pipeline.lectura.return_value = (pipeline.array, preview)
    # Act
    resultado = integrator.analyze("entrada.dcm")
    # Assert
    assert resultado["preview"] is preview


def test_analyze_invoca_modelo_exactamente_una_vez(integrator, pipeline, fake_model):
    # Arrange
    ruta = "entrada.jpg"
    # Act
    integrator.analyze(ruta)
    # Assert
    assert fake_model.llamadas == 1


def test_analyze_entrega_al_modelo_el_tensor_preprocesado(integrator, pipeline, fake_model):
    # Arrange
    esperado = pipeline.tensor
    # Act
    integrator.analyze("entrada.jpg")
    # Assert
    assert fake_model.ultima_entrada is esperado


def test_analyze_entrega_al_preprocesador_el_array_del_lector(integrator, pipeline):
    # Arrange
    esperado = pipeline.array
    # Act
    integrator.analyze("entrada.jpg")
    # Assert
    pipeline.preprocesar.assert_called_once_with(esperado)


def test_analyze_respeta_orden_lectura_preprocesamiento_prediccion(
    integrator, pipeline, fake_model, monkeypatch
):
    # Arrange
    eventos = []

    def leer(ruta):
        eventos.append("lectura")
        return pipeline.array, pipeline.preview

    def preprocesar(array):
        eventos.append("preprocesamiento")
        return pipeline.tensor

    predecir_original = fake_model.predict

    def predecir(tensor):
        eventos.append("prediccion")
        return predecir_original(tensor)

    pipeline.lectura.side_effect = leer
    pipeline.preprocesar.side_effect = preprocesar
    monkeypatch.setattr(fake_model, "predict", predecir)
    # Act
    integrator.analyze("entrada.jpg")
    # Assert
    assert eventos == ["lectura", "preprocesamiento", "prediccion"]


def test_read_file_delega_en_la_capa_de_lectura(integrator, monkeypatch):
    # Arrange — el despacho por extensión vive en read_img (alta cohesión).
    esperado = (np.zeros((2, 2, 3), dtype=np.uint8), Image.new("RGB", (2, 2)))
    lector = Mock(return_value=esperado)
    monkeypatch.setattr(modulo_integrator, "read_file", lector)
    # Act
    salida = integrator._read_file("rx.dcm")
    # Assert
    assert salida is esperado
    lector.assert_called_once_with("rx.dcm")


def test_supported_extensions_se_expone_desde_el_controlador():
    # Arrange / Act / Assert — la vista lee los formatos sin importar la capa modelo.
    from src.models.read_img import SUPPORTED_EXTENSIONS

    assert modulo_integrator.SUPPORTED_EXTENSIONS is SUPPORTED_EXTENSIONS


def test_analyze_rechaza_dicom_inexistente(integrator, tmp_path):
    # Arrange
    ruta = str(tmp_path / "inexistente.dcm")
    # Act
    with pytest.raises(FileNotFoundError, match="No se encontró el archivo") as error:
        integrator.analyze(ruta)
    # Assert
    assert str(error.value) == f"No se encontró el archivo: {ruta}"


def test_analyze_rechaza_jpg_inexistente(integrator, tmp_path):
    # Arrange
    ruta = str(tmp_path / "inexistente.jpg")
    # Act
    with pytest.raises(FileNotFoundError, match="No se encontró el archivo") as error:
        integrator.analyze(ruta)
    # Assert
    assert str(error.value) == f"No se encontró el archivo: {ruta}"


def test_analyze_no_invoca_modelo_si_archivo_no_existe(integrator, fake_model, tmp_path):
    # Arrange
    ruta = str(tmp_path / "inexistente.jpg")
    # Act
    with pytest.raises(FileNotFoundError, match="No se encontró el archivo"):
        integrator.analyze(ruta)
    # Assert
    assert fake_model.llamadas == 0


def test_analyze_propaga_error_del_lector(integrator, ruta_jpg, monkeypatch):
    # Arrange
    esperado = FileNotFoundError(f"No se pudo leer la imagen: {ruta_jpg}")
    monkeypatch.setattr(modulo_integrator, "read_file", Mock(side_effect=esperado))
    # Act
    with pytest.raises(FileNotFoundError, match="No se pudo leer la imagen") as error:
        integrator.analyze(ruta_jpg)
    # Assert
    assert error.value is esperado


def test_analyze_propaga_error_de_preprocesamiento(integrator, pipeline, monkeypatch):
    # Arrange — se integra el preprocesador real solo para verificar propagación.
    pipeline.lectura.return_value = (np.array([], dtype=np.uint8), pipeline.preview)
    monkeypatch.setattr(modulo_integrator, "preprocess", preprocess)
    # Act
    with pytest.raises(ValueError, match="vacío") as error:
        integrator.analyze("entrada.jpg")
    # Assert
    assert str(error.value) == "El array de entrada está vacío"


def test_analyze_propaga_error_del_modelo(integrator, pipeline, fake_model, monkeypatch):
    # Arrange
    esperado = RuntimeError("Fallo simulado de inferencia")
    monkeypatch.setattr(fake_model, "predict", Mock(side_effect=esperado))
    # Act
    with pytest.raises(RuntimeError, match="Fallo simulado de inferencia") as error:
        integrator.analyze("entrada.jpg")
    # Assert
    assert error.value is esperado


def test_generate_report_delega_en_el_modelo_de_reporte(integrator, monkeypatch):
    # Arrange
    from datetime import datetime

    constructor = Mock(return_value=b"%PDF-fake")
    monkeypatch.setattr(modulo_integrator, "build_report_pdf", constructor)
    resultado = {"label": "viral", "probabilities": [0.1, 0.2, 0.7], "preview": Mock()}
    fecha = datetime(2026, 10, 29, 9, 0, 0)
    # Act
    contenido = integrator.generate_report("Ana Pérez", "123", resultado, fecha, ".dcm")
    # Assert
    assert contenido == b"%PDF-fake"
    constructor.assert_called_once_with("Ana Pérez", "123", "viral", [0.1, 0.2, 0.7], fecha, ".dcm")


def test_report_filename_se_expone_desde_el_controlador():
    # Arrange
    from datetime import datetime

    # Act
    nombre = modulo_integrator.report_filename("123", datetime(2026, 10, 29, 9, 0, 0))
    # Assert
    assert nombre == "informe_123_20261029_090000.pdf"
