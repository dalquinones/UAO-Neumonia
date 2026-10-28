"""Pruebas AAA de la explicabilidad de Integrator (explain + fábrica), sin TensorFlow.

El controlador compone lectura + preprocesamiento + predicción con Grad-CAM y
entrega el overlay convertido a PIL RGB, para que la vista nunca importe cv2 ni
tensorflow. El modelo y el backend de TF se simulan por completo.
"""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from PIL import Image

from src.controllers import integrator as modulo_integrador
from src.controllers.integrator import Integrator, crear_integrador


class _TensorSimulado(np.ndarray):
    """Array NumPy con la conversión .numpy() usada por Grad-CAM."""

    def numpy(self) -> np.ndarray:
        """Devuelve los valores sin ejecutar TensorFlow."""
        return np.asarray(self)


def _tensor(valor) -> _TensorSimulado:
    """Construye un tensor float32 para la aritmética de producción."""
    return np.asarray(valor, dtype=np.float32).view(_TensorSimulado)


@pytest.fixture(autouse=True)
def backend_tf(monkeypatch):
    """Inyecta un backend TensorFlow NumPy con gradientes programados."""
    activaciones = _tensor(np.arange(1, 9).reshape(1, 2, 2, 2))
    gradientes = _tensor(np.ones((1, 2, 2, 2)))
    predicciones = _tensor([[0.05, 0.90, 0.05]])
    capa = SimpleNamespace(output=object())
    modelo = SimpleNamespace(
        get_layer=Mock(return_value=capa),
        inputs=object(),
        output=object(),
    )
    modelo.predict = Mock(return_value=predicciones)

    submodelo = Mock(side_effect=lambda entrada: (activaciones, predicciones))
    cinta = MagicMock()
    cinta.__enter__.return_value = cinta
    cinta.__exit__.return_value = False
    cinta.gradient.side_effect = lambda perdida, conv_output: gradientes

    modulo = ModuleType("tensorflow")
    modulo.keras = SimpleNamespace(Model=Mock(return_value=submodelo))
    modulo.GradientTape = Mock(return_value=cinta)
    modulo.reduce_mean = lambda valor, axis: _tensor(np.mean(valor, axis=axis))
    modulo.newaxis = np.newaxis
    modulo.maximum = lambda valor, minimo: _tensor(np.maximum(valor, minimo))
    modulo.squeeze = lambda valor, axis=None: _tensor(np.squeeze(valor, axis=axis))
    modulo.math = SimpleNamespace(reduce_max=np.max)
    modulo.equal = np.equal
    modulo.zeros_like = lambda valor: _tensor(np.zeros_like(valor))
    monkeypatch.setitem(sys.modules, "tensorflow", modulo)

    return SimpleNamespace(
        modelo=modelo,
        capa=capa,
        activaciones=activaciones,
        gradientes=gradientes,
        predicciones=predicciones,
        submodelo=submodelo,
        cinta=cinta,
        backend=modulo,
    )


@pytest.fixture
def controlador(backend_tf) -> Integrator:
    """Controlador con el modelo simulado (predict + get_layer + inputs/output)."""
    return Integrator(model=backend_tf.modelo)


def test_explain_devuelve_claves_exactas(controlador, ruta_jpg):
    # Arrange
    claves = {"label", "probabilities", "preview", "overlay"}
    # Act
    resultado = controlador.explain(ruta_jpg)
    # Assert
    assert isinstance(resultado, dict) and set(resultado) == claves


def test_explain_predice_etiqueta_normal_con_modelo_programado(controlador, ruta_jpg):
    # Arrange — el modelo simulado favorece la clase 1.
    # Act
    resultado = controlador.explain(ruta_jpg)
    # Assert
    assert resultado["label"] == "normal"
    assert resultado["probabilities"] == pytest.approx([0.05, 0.90, 0.05])


def test_explain_entrega_overlay_pil_rgb(controlador, ruta_jpg):
    # Arrange
    # Act
    resultado = controlador.explain(ruta_jpg)
    # Assert — la vista recibe una imagen PIL RGB lista para pintar.
    assert isinstance(resultado["overlay"], Image.Image)
    assert resultado["overlay"].mode == "RGB"


def test_explain_invoca_modelo_exactamente_una_vez(controlador, backend_tf, ruta_jpg):
    # Arrange
    # Act
    controlador.explain(ruta_jpg)
    # Assert
    backend_tf.modelo.predict.assert_called_once()


def test_explain_usa_la_clase_ganadora_para_gradcam(controlador, backend_tf, ruta_jpg):
    # Arrange — la clase ganadora programada es la 1 (0.90).
    # Act
    controlador.explain(ruta_jpg)
    # Assert — la pérdida de Grad-CAM corresponde a la clase ganadora.
    perdida = backend_tf.cinta.gradient.call_args.args[0]
    np.testing.assert_allclose(perdida, [0.90])


def test_explain_transmite_mismo_tensor_a_predict_y_gradcam(controlador, backend_tf, ruta_jpg):
    # Arrange
    # Act
    controlador.explain(ruta_jpg)
    # Assert — el subgrafo Grad-CAM recibe el mismo tensor que predict.
    tensor_predict = backend_tf.modelo.predict.call_args.args[0]
    assert backend_tf.submodelo.call_args.args[0] is tensor_predict


def test_explain_dicom_genera_overlay_rgb(controlador, ruta_dcm):
    # Arrange
    # Act
    resultado = controlador.explain(ruta_dcm)
    # Assert — el recorrido DICOM real también termina en overlay RGB.
    assert resultado["label"] == "normal"
    assert isinstance(resultado["overlay"], Image.Image)
    assert resultado["overlay"].mode == "RGB"


def test_explain_rechaza_archivo_inexistente(controlador, backend_tf, tmp_path):
    # Arrange
    ruta = str(tmp_path / "no_existe.jpg")
    # Act
    with pytest.raises(FileNotFoundError, match="No se encontró el archivo") as error:
        controlador.explain(ruta)
    # Assert — no se alcanza la inferencia.
    assert str(error.value) == f"No se encontró el archivo: {ruta}"
    backend_tf.modelo.predict.assert_not_called()


def test_explain_propaga_error_de_preprocesamiento(controlador, backend_tf, monkeypatch):
    # Arrange — array vacío para activar la validación del preprocesador.
    preview = Image.new("RGB", (2, 2))
    monkeypatch.setattr(
        controlador, "_read_file", lambda ruta: (np.array([], dtype=np.uint8), preview)
    )
    # Act
    with pytest.raises(ValueError, match="vacío"):
        controlador.explain("entrada.jpg")
    # Assert
    backend_tf.modelo.predict.assert_not_called()


def test_explain_propaga_error_del_modelo(controlador, backend_tf, ruta_jpg):
    # Arrange
    causa = RuntimeError("fallo de inferencia simulado")
    backend_tf.modelo.predict = Mock(side_effect=causa)
    # Act
    with pytest.raises(RuntimeError, match="fallo de inferencia simulado") as error:
        controlador.explain(ruta_jpg)
    # Assert
    assert error.value is causa


def test_explain_informa_modelo_sin_capa_gradcam(controlador, backend_tf, ruta_jpg):
    # Arrange — el modelo no expone la capa conv requerida.
    backend_tf.modelo.get_layer.side_effect = ValueError("capa ausente")
    # Act
    with pytest.raises(ValueError, match="Grad-CAM necesita la capa 'conv10_thisone'"):
        controlador.explain(ruta_jpg)
    # Assert — el diagnóstico llega desde el modelo y la capa fue consultada.
    backend_tf.modelo.get_layer.assert_called_once_with("conv10_thisone")


def test_crear_integrador_inyecta_el_modelo_cargado(monkeypatch):
    # Arrange — cargador simulado que evita TensorFlow real.
    modelo = object()
    cargador = Mock(return_value=modelo)
    monkeypatch.setattr("src.models.load_model.load_model", cargador)
    # Act
    controlador = crear_integrador()
    # Assert
    assert isinstance(controlador, Integrator)
    assert controlador.model is modelo
    cargador.assert_called_once_with(None)


def test_crear_integrador_propaga_falta_de_checkpoint(monkeypatch):
    # Arrange — sin checkpoint, la carga falla y no hay nada que inyectar.
    causa = FileNotFoundError("No se encontró el archivo del modelo")
    monkeypatch.setattr("src.models.load_model.load_model", Mock(side_effect=causa))
    # Act
    with pytest.raises(FileNotFoundError, match="No se encontró el archivo del modelo") as error:
        crear_integrador()
    # Assert
    assert error.value is causa


def test_explain_reutiliza_mismo_preprocesamiento_que_analyze(controlador, ruta_jpg, monkeypatch):
    # Arrange — espiar el preprocesador compartido por ambos métodos.
    procesador = Mock(wraps=modulo_integrador.preprocess)
    monkeypatch.setattr(modulo_integrador, "preprocess", procesador)
    # Act
    controlador.explain(ruta_jpg)
    # Assert
    procesador.assert_called_once()
