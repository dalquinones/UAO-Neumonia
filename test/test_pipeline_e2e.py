"""Once escenarios del controlador y cinco composiciones controladas, todos AAA.

Lectores, preprocesamiento y controlador son reales; la inferencia es simulada.
Dos escenarios del controlador inyectan fallos en CLAHE o predict. Las cinco
composiciones conectan explícitamente carga o Grad-CAM: analyze no lo hace.
El defecto (h, w, 1) de Grad-CAM se reproduce, sin corregirlo ni certificar 2D.
"""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import cv2
import numpy as np
import pytest
from PIL import Image
from pydicom.errors import InvalidDicomError

from src.controllers import integrator as modulo_integrador
from src.controllers.integrator import Integrator
from src.models import preprocess_img as modulo_preprocesamiento
from src.models.grad_cam import make_gradcam_heatmap, overlay_heatmap
from src.models.load_model import load_model
from src.models.preprocess_img import preprocess
from src.models.read_img import read_dicom, read_image


class _TensorSimulado(np.ndarray):
    """Array NumPy con la conversión usada por el algoritmo de Grad-CAM."""

    def numpy(self) -> np.ndarray:
        """Devuelve un array sin ejecutar TensorFlow."""
        return np.asarray(self)


def _tensor(valor) -> _TensorSimulado:
    """Convierte datos a un tensor simulado float32."""
    return np.asarray(valor, dtype=np.float32).view(_TensorSimulado)


def _modelo_espia(fabrica, clase=0):
    """Conserva copias de las entradas y el comportamiento del FakeModel compartido."""
    modelo = fabrica(clase_programada=clase)
    predecir_original = modelo.predict
    modelo.entradas = []

    def predecir(tensor):
        modelo.entradas.append(tensor.copy())
        return predecir_original(tensor)

    modelo.predict = Mock(side_effect=predecir)
    return modelo


def _comprobar_tensor(tensor) -> None:
    """Comprueba el contrato en la frontera entre procesamiento e inferencia."""
    assert tensor.shape == (1, 512, 512, 1)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0


def _comprobar_resultado(resultado, etiqueta, probabilidades) -> None:
    """Comprueba el contrato público, sin exigir igualdad binaria entre floats."""
    assert set(resultado) == {"label", "probabilities", "preview"}
    assert resultado["label"] == etiqueta
    assert len(resultado["probabilities"]) == 3
    assert all(type(valor) is float for valor in resultado["probabilities"])
    np.testing.assert_allclose(resultado["probabilities"], probabilidades)
    assert isinstance(resultado["preview"], Image.Image)


def _cargar_y_analizar(ruta_imagen, ruta_modelo=None):
    """Composición exclusiva de pruebas: la carga ocurre fuera de analyze."""
    modelo = load_model(ruta_modelo)
    return Integrator(modelo).analyze(ruta_imagen)


def _analizar_y_generar_mapa(modelo, ruta_imagen):
    """Conecta explícitamente la inferencia con Grad-CAM, fuera del controlador."""
    resultado = Integrator(modelo).analyze(ruta_imagen)
    mapa = make_gradcam_heatmap(modelo, modelo.ultima_entrada)
    return resultado, mapa


def _pipeline_con_superposicion(ruta_modelo, ruta_imagen, registro):
    """Compone funciones existentes y registra avances antes del defecto conocido."""
    modelo = load_model(ruta_modelo)
    registro.resultado, registro.mapa = _analizar_y_generar_mapa(modelo, ruta_imagen)
    array, _ = read_image(ruta_imagen)
    return overlay_heatmap(registro.mapa, array)


@pytest.fixture(autouse=True)
def backend_simulado(monkeypatch, fakemodel_factory, fake_net):
    """Aísla TensorFlow y ofrece un modelo compatible con carga, predict y Grad-CAM.

    Los gradientes son programados: no se verifica autodiferenciación real.
    El parche se restaura automáticamente después de cada prueba.
    """
    modelo = _modelo_espia(fakemodel_factory, clase=1)
    modelo.layers = fake_net.layers
    modelo.output_shape = fake_net.output_shape
    capa = SimpleNamespace(output=object())
    modelo.get_layer = Mock(return_value=capa)
    modelo.inputs = object()
    modelo.output = object()
    escenario = SimpleNamespace(
        modelo=modelo,
        capa=capa,
        activaciones=_tensor(np.arange(1, 9).reshape(1, 2, 2, 2)),
        gradientes=_tensor(np.ones((1, 2, 2, 2))),
        predicciones=_tensor([[0.05, 0.90, 0.05]]),
    )
    escenario.submodelo = Mock(
        side_effect=lambda entrada: (escenario.activaciones, escenario.predicciones)
    )
    escenario.cinta = MagicMock()
    escenario.cinta.__enter__.return_value = escenario.cinta
    escenario.cinta.__exit__.return_value = False
    escenario.cinta.gradient.side_effect = lambda perdida, activaciones: escenario.gradientes
    escenario.cargador = Mock(return_value=modelo)
    modulo = ModuleType("tensorflow")
    modulo.keras = SimpleNamespace(
        models=SimpleNamespace(load_model=escenario.cargador),
        Model=Mock(return_value=escenario.submodelo),
    )
    modulo.GradientTape = Mock(return_value=escenario.cinta)
    modulo.reduce_mean = lambda valor, axis: _tensor(np.mean(valor, axis=axis))
    modulo.newaxis = np.newaxis
    modulo.maximum = lambda valor, minimo: _tensor(np.maximum(valor, minimo))
    modulo.math = SimpleNamespace(reduce_max=np.max)
    modulo.equal = np.equal
    modulo.zeros_like = lambda valor: _tensor(np.zeros_like(valor))
    monkeypatch.setitem(sys.modules, "tensorflow", modulo)
    escenario.backend = modulo
    return escenario


@pytest.fixture
def ruta_checkpoint(tmp_path):
    """Archivo ficticio: solo satisface la comprobación de existencia, nunca es leído."""
    ruta = tmp_path / "modelo_simulado.h5"
    ruta.write_bytes(b"checkpoint ficticio para cargador simulado")
    return str(ruta)


# Integración del recorrido nativo: lectores y preprocesamiento reales, modelo doble.


def test_pipeline_jpg_entrega_resultado_bacteriano_y_tensor_valido(ruta_jpg, fakemodel_factory):
    # Arrange — imagen real, modelo doble y dimensiones originales como referencia.
    modelo = _modelo_espia(fakemodel_factory, clase=0)
    with Image.open(ruta_jpg) as original:
        dimensiones = original.size

    # Act — recorrer el pipeline público del controlador.
    resultado = Integrator(modelo).analyze(ruta_jpg)

    # Assert — se respeta el contrato desde archivo hasta resultado e inferencia.
    _comprobar_resultado(resultado, "bacteriana", [0.90, 0.05, 0.05])
    assert resultado["preview"].mode == "RGB"
    assert resultado["preview"].size == dimensiones
    modelo.predict.assert_called_once()
    _comprobar_tensor(modelo.ultima_entrada)


def test_pipeline_dicom_entrega_resultado_normal_y_preview_gris(ruta_dcm, fakemodel_factory):
    # Arrange — DICOM real y referencia del lector, sin sustituirlo en el controlador.
    modelo = _modelo_espia(fakemodel_factory, clase=1)
    _, referencia = read_dicom(ruta_dcm)

    # Act — analizar el DICOM con lectura y procesamiento reales.
    resultado = Integrator(modelo).analyze(ruta_dcm)

    # Assert — la normalización DICOM alcanza el modelo y conserva la preview gris.
    _comprobar_resultado(resultado, "normal", [0.05, 0.90, 0.05])
    assert resultado["preview"].mode == "L"
    np.testing.assert_array_equal(np.asarray(resultado["preview"]), np.asarray(referencia))
    modelo.predict.assert_called_once()
    _comprobar_tensor(modelo.ultima_entrada)


def test_pipeline_jpg_conserva_orden_de_probabilidades_para_clase_viral(
    ruta_jpg, fakemodel_factory
):
    # Arrange — programar la tercera clase sin alterar lectores ni preprocesamiento.
    modelo = _modelo_espia(fakemodel_factory, clase=2)

    # Act — analizar la imagen real.
    resultado = Integrator(modelo).analyze(ruta_jpg)

    # Assert — el controlador conserva el orden bacteriana, normal, viral.
    _comprobar_resultado(resultado, "viral", [0.05, 0.05, 0.90])
    modelo.predict.assert_called_once()
    _comprobar_tensor(modelo.ultima_entrada)


def test_pipeline_repetido_conserva_tensor_prediccion_y_preview(ruta_jpg, fakemodel_factory):
    # Arrange — reutilizar un controlador y registrar copias de cada tensor.
    modelo = _modelo_espia(fakemodel_factory)
    controlador = Integrator(modelo)

    # Act — ejecutar el escenario de dos análisis consecutivos del mismo archivo.
    primero, segundo = [controlador.analyze(ruta_jpg) for _ in range(2)]

    # Assert — comparar contenidos, no identidad de objetos PIL.
    assert modelo.predict.call_count == 2
    np.testing.assert_array_equal(modelo.entradas[0], modelo.entradas[1])
    assert primero["label"] == segundo["label"]
    assert primero["probabilities"] == segundo["probabilities"]
    np.testing.assert_array_equal(np.asarray(primero["preview"]), np.asarray(segundo["preview"]))


def test_pipeline_alterna_dicom_y_jpg_sin_contaminar_resultados(
    ruta_dcm, ruta_jpg, fakemodel_factory
):
    # Arrange — obtener referencias independientes para cada lector y tensor.
    array_dcm, preview_dcm = read_dicom(ruta_dcm)
    array_jpg, preview_jpg = read_image(ruta_jpg)
    esperados = [preprocess(array_dcm), preprocess(array_jpg)]
    modelo = _modelo_espia(fakemodel_factory)
    controlador = Integrator(modelo)

    # Act — alternar formatos sobre el mismo controlador.
    resultados = [controlador.analyze(ruta) for ruta in (ruta_dcm, ruta_jpg)]

    # Assert — cada llamada conserva el contenido del archivo que le corresponde.
    assert modelo.predict.call_count == 2
    for entrada, esperado in zip(modelo.entradas, esperados, strict=True):
        np.testing.assert_array_equal(entrada, esperado)
    for resultado, preview in zip(resultados, (preview_dcm, preview_jpg), strict=True):
        assert resultado["preview"].mode == preview.mode
        np.testing.assert_array_equal(np.asarray(resultado["preview"]), np.asarray(preview))


def test_pipeline_resuelve_empate_de_predicciones_con_primera_clase(ruta_jpg, fake_model):
    # Arrange — predicciones empatadas sin modificar el recorrido del archivo.
    fake_model.predict = Mock(return_value=np.full((1, 3), 1 / 3, dtype=np.float32))

    # Act — analizar una imagen con empate de las tres clases.
    resultado = Integrator(fake_model).analyze(ruta_jpg)

    # Assert — argmax elige la primera clase y el resultado sigue siendo utilizable.
    _comprobar_resultado(resultado, "bacteriana", [1 / 3, 1 / 3, 1 / 3])
    fake_model.predict.assert_called_once()
    _comprobar_tensor(fake_model.predict.call_args.args[0])


def test_pipeline_ruta_inexistente_impide_preprocesamiento_e_inferencia(
    tmp_path, monkeypatch, fake_model
):
    # Arrange — vigilar las etapas posteriores sin sustituir su comportamiento.
    ruta = str(tmp_path / "no_existe.jpg")
    lector_jpg = Mock(wraps=read_image)
    lector_dcm = Mock(wraps=read_dicom)
    procesador = Mock(wraps=preprocess)
    monkeypatch.setattr(modulo_integrador, "read_image", lector_jpg)
    monkeypatch.setattr(modulo_integrador, "read_dicom", lector_dcm)
    monkeypatch.setattr(modulo_integrador, "preprocess", procesador)

    # Act — analizar una ruta que no existe.
    with pytest.raises(FileNotFoundError, match="No se encontró el archivo") as error:
        Integrator(fake_model).analyze(ruta)

    # Assert — no se alcanza ninguna etapa posterior a la comprobación de existencia.
    assert ruta in str(error.value)
    lector_jpg.assert_not_called()
    lector_dcm.assert_not_called()
    procesador.assert_not_called()
    assert fake_model.llamadas == 0


def test_pipeline_imagen_corrupta_propaga_error_sin_inferencia(tmp_path, fake_model):
    # Arrange — archivo existente que el decodificador real no puede leer.
    ruta = tmp_path / "corrupta.jpg"
    ruta.write_bytes(b"esto no es una imagen")

    # Act — atravesar controlador y lector real.
    with pytest.raises(FileNotFoundError, match="No se pudo leer la imagen") as error:
        Integrator(fake_model).analyze(str(ruta))

    # Assert — la causa de lectura se informa sin ejecutar inferencia.
    assert str(ruta) in str(error.value)
    assert fake_model.llamadas == 0


def test_pipeline_dicom_corrupto_propaga_error_del_parser(tmp_path, fake_model):
    # Arrange — contenido sin cabecera DICOM para ejercitar pydicom realmente.
    ruta = tmp_path / "corrupto.dcm"
    ruta.write_bytes(b"esto no es un DICOM")

    # Act — dejar que la excepción específica atraviese el controlador.
    with pytest.raises(InvalidDicomError, match="DICM") as error:
        Integrator(fake_model).analyze(str(ruta))

    # Assert — el error no se convierte en un resultado ni en una inferencia.
    assert isinstance(error.value, InvalidDicomError)
    assert fake_model.llamadas == 0


def test_pipeline_fallo_de_clahe_impide_inferencia(ruta_jpg, monkeypatch, fake_model):
    # Arrange — inyectar un fallo únicamente en la dependencia CLAHE del preprocesador.
    causa = cv2.error("fallo CLAHE simulado")
    clahe = Mock(side_effect=causa)
    lector = Mock(wraps=read_image)
    monkeypatch.setattr(modulo_preprocesamiento.cv2, "createCLAHE", clahe)
    monkeypatch.setattr(modulo_integrador, "read_image", lector)

    # Act — leer y procesar hasta alcanzar el fallo programado.
    with pytest.raises(cv2.error, match="fallo CLAHE simulado") as error:
        Integrator(fake_model).analyze(ruta_jpg)

    # Assert — se leyó el archivo, pero la inferencia quedó bloqueada.
    assert error.value is causa
    lector.assert_called_once_with(ruta_jpg)
    clahe.assert_called_once()
    assert fake_model.llamadas == 0


def test_pipeline_propaga_error_del_modelo_tras_preprocesar_imagen(ruta_jpg, fake_model):
    # Arrange — fallo de inferencia después de lectura y preprocesamiento reales.
    causa = RuntimeError("fallo de inferencia simulado")
    fake_model.predict = Mock(side_effect=causa)

    # Act — ejecutar el análisis hasta la frontera neuronal.
    with pytest.raises(RuntimeError, match="fallo de inferencia simulado") as error:
        Integrator(fake_model).analyze(ruta_jpg)

    # Assert — se preserva la causa y el modelo recibió una entrada válida.
    assert error.value is causa
    fake_model.predict.assert_called_once()
    _comprobar_tensor(fake_model.predict.call_args.args[0])


# Composiciones controladas: analyze no carga modelos ni invoca Grad-CAM.


def test_pipeline_carga_modelo_del_entorno_y_analiza_imagen_real(
    ruta_jpg, ruta_checkpoint, monkeypatch, backend_simulado
):
    # Arrange — ruta del entorno y cargador ficticio con arquitectura validable real.
    escenario = backend_simulado
    monkeypatch.setenv("MODEL_PATH", ruta_checkpoint)

    # Act — componer carga externa y análisis, sin atribuir la carga a analyze.
    resultado = _cargar_y_analizar(ruta_jpg)

    # Assert — el modelo cargado y validado es el que realiza la inferencia.
    escenario.cargador.assert_called_once_with(ruta_checkpoint)
    escenario.modelo.predict.assert_called_once()
    _comprobar_tensor(escenario.modelo.ultima_entrada)
    _comprobar_resultado(resultado, "normal", [0.05, 0.90, 0.05])


def test_pipeline_arquitectura_invalida_bloquea_analisis(
    ruta_jpg, ruta_checkpoint, monkeypatch, backend_simulado
):
    # Arrange — la validación de producción debe rechazar la capa ausente.
    escenario = backend_simulado
    escenario.modelo.layers = [SimpleNamespace(name="dense_final")]
    analisis = Mock()
    monkeypatch.setattr(Integrator, "analyze", analisis)

    # Act — intentar la composición con carga antes de analizar.
    with pytest.raises(ValueError, match="conv10_thisone"):
        _cargar_y_analizar(ruta_jpg, ruta_checkpoint)

    # Assert — la barrera de arquitectura impide llegar al análisis y a predict.
    escenario.cargador.assert_called_once_with(ruta_checkpoint)
    analisis.assert_not_called()
    escenario.modelo.predict.assert_not_called()


def test_pipeline_error_de_carga_impide_lectura_e_inferencia(
    ruta_jpg, ruta_checkpoint, monkeypatch, backend_simulado
):
    # Arrange — fallo de deserialización programado en el cargador externo.
    escenario = backend_simulado
    causa = OSError("checkpoint corrupto simulado")
    escenario.cargador.side_effect = causa
    lector = Mock(wraps=read_image)
    analisis = Mock()
    monkeypatch.setattr(modulo_integrador, "read_image", lector)
    monkeypatch.setattr(Integrator, "analyze", analisis)

    # Act — intentar inicializar el modelo antes del recorrido del archivo.
    with pytest.raises(OSError, match="checkpoint corrupto simulado") as error:
        _cargar_y_analizar(ruta_jpg, ruta_checkpoint)

    # Assert — se conserva el error y ninguna etapa de análisis comienza.
    assert error.value is causa
    escenario.cargador.assert_called_once_with(ruta_checkpoint)
    analisis.assert_not_called()
    lector.assert_not_called()
    escenario.modelo.predict.assert_not_called()


def test_pipeline_dicom_transmite_tensor_real_a_gradcam_con_gradientes_nulos(
    ruta_dcm, backend_simulado
):
    # Arrange — DICOM real y gradientes nulos para la composición explícita.
    escenario = backend_simulado
    escenario.gradientes = _tensor(np.zeros((1, 2, 2, 2)))

    # Act — analizar y entregar a Grad-CAM el mismo tensor usado en predict.
    resultado, mapa = _analizar_y_generar_mapa(escenario.modelo, ruta_dcm)

    # Assert — se transmite el tensor por identidad; la forma 3D sigue siendo un defecto.
    _comprobar_resultado(resultado, "normal", [0.05, 0.90, 0.05])
    _comprobar_tensor(escenario.modelo.ultima_entrada)
    escenario.modelo.predict.assert_called_once()
    escenario.submodelo.assert_called_once()
    assert escenario.submodelo.call_args.args[0] is escenario.modelo.ultima_entrada
    escenario.modelo.get_layer.assert_called_once_with("conv10_thisone")
    assert mapa.shape == (2, 2, 1)  # Incumple el contrato documentado (h, w).
    np.testing.assert_array_equal(mapa, np.zeros_like(mapa))
    assert np.isfinite(mapa).all()
    escenario.cargador.assert_not_called()  # La carga no pertenece a analyze.


def test_pipeline_completo_reproduce_incompatibilidad_dimensional_de_gradcam(
    ruta_jpg, ruta_checkpoint, backend_simulado
):
    # Arrange — modelo cargable simulado, imagen real y registro de etapas anteriores.
    escenario = backend_simulado
    registro = SimpleNamespace()
    esperado = np.array([[[3], [7]], [[11], [15]]], dtype=np.float32) / 15

    # Act — componer carga, análisis, Grad-CAM y overlay sin eliminar el eje final.
    with pytest.raises(ValueError, match=r"mapa debe ser 2D.*ndim=3") as error:
        _pipeline_con_superposicion(ruta_checkpoint, ruta_jpg, registro)

    # Assert — las etapas previas funcionan; el defecto bloquea la superposición directa.
    escenario.cargador.assert_called_once_with(ruta_checkpoint)
    escenario.modelo.predict.assert_called_once()
    _comprobar_resultado(registro.resultado, "normal", [0.05, 0.90, 0.05])
    _comprobar_tensor(escenario.modelo.ultima_entrada)
    escenario.submodelo.assert_called_once()
    assert escenario.submodelo.call_args.args[0] is escenario.modelo.ultima_entrada
    escenario.modelo.get_layer.assert_called_once_with("conv10_thisone")
    np.testing.assert_allclose(registro.mapa, esperado)
    assert registro.mapa.shape == (2, 2, 1)  # Defecto reproducido, no corregido.
    assert registro.mapa.dtype == np.float32
    assert np.isfinite(registro.mapa).all()
    assert 0 <= registro.mapa.min() <= registro.mapa.max() == 1
    assert "recibí ndim=3" in str(error.value)
