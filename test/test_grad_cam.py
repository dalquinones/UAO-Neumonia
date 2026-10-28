"""Pruebas AAA de Grad-CAM con TensorFlow simulado y OpenCV real.

La salida de make_gradcam_heatmap cumple el contrato documentado (h, w): el
eje final unitario se elimina con squeeze y el mapa puede superponerse
directamente con overlay_heatmap.
"""

import builtins
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import cv2
import numpy as np
import pytest

from src.models.grad_cam import make_gradcam_heatmap, overlay_heatmap


class _TensorSimulado(np.ndarray):
    """Array con operaciones NumPy y conversión .numpy() compatible con el módulo."""

    def numpy(self) -> np.ndarray:
        """Entrega los valores sin depender de TensorFlow."""
        return np.asarray(self)


def _tensor(valor) -> _TensorSimulado:
    """Construye un tensor float32 para ejecutar la aritmética de producción."""
    return np.asarray(valor, dtype=np.float32).view(_TensorSimulado)


@pytest.fixture(autouse=True)
def gradcam_simulado(monkeypatch):
    """Inyecta un backend NumPy con gradientes programados, sin autodiferenciación."""
    escenario = SimpleNamespace(
        activaciones=_tensor(np.arange(1, 9).reshape(1, 2, 2, 2)),
        gradientes=_tensor(np.ones((1, 2, 2, 2))),
        predicciones=_tensor([[0.1, 0.7, 0.2]]),
        tensor=np.zeros((1, 4, 6, 1), dtype=np.float32),
    )
    escenario.capa = SimpleNamespace(output=object())
    escenario.modelo = SimpleNamespace(
        get_layer=Mock(return_value=escenario.capa), inputs=object(), output=object()
    )
    escenario.submodelo = Mock(
        side_effect=lambda tensor: (escenario.activaciones, escenario.predicciones)
    )
    escenario.cinta = MagicMock()
    escenario.cinta.__enter__.return_value = escenario.cinta
    escenario.cinta.__exit__.return_value = False
    escenario.cinta.gradient.side_effect = lambda loss, conv_output: escenario.gradientes

    modulo = ModuleType("tensorflow")
    modulo.keras = SimpleNamespace(Model=Mock(return_value=escenario.submodelo))
    modulo.GradientTape = Mock(return_value=escenario.cinta)
    modulo.reduce_mean = Mock(side_effect=lambda valor, axis: _tensor(np.mean(valor, axis=axis)))
    modulo.newaxis = np.newaxis
    modulo.maximum = lambda valor, minimo: _tensor(np.maximum(valor, minimo))
    modulo.math = SimpleNamespace(reduce_max=np.max)
    modulo.equal = np.equal
    modulo.zeros_like = lambda valor: _tensor(np.zeros_like(valor))
    modulo.squeeze = lambda valor, axis=None: _tensor(np.squeeze(valor, axis=axis))
    monkeypatch.setitem(sys.modules, "tensorflow", modulo)
    escenario.backend = modulo
    return escenario


def test_importar_modulo_no_importa_tensorflow(monkeypatch):
    # Arrange — vigilar incluso los intentos de importar el módulo ficticio.
    importar_original = builtins.__import__
    intentos = Mock()

    def importar(nombre, globals=None, locals=None, fromlist=(), level=0):
        if nombre == "tensorflow" or nombre.startswith("tensorflow."):
            intentos(nombre)
            raise AssertionError("No se debe importar TensorFlow al cargar el módulo")
        return importar_original(nombre, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", importar)
    ruta = Path(__file__).resolve().parents[1] / "src" / "models" / "grad_cam.py"

    # Act — ejecutar las definiciones en un espacio de nombres aislado.
    espacio = runpy.run_path(str(ruta))

    # Assert — ambas funciones están disponibles sin intentar importar TensorFlow.
    assert callable(espacio["make_gradcam_heatmap"])
    assert callable(espacio["overlay_heatmap"])
    intentos.assert_not_called()


def test_gradcam_utiliza_conv10_thisone_y_transmite_tensor_al_submodelo(gradcam_simulado):
    # Arrange — identificar las entradas, capa y salidas del modelo ficticio.
    escenario = gradcam_simulado

    # Act — generar el mapa mediante el submodelo espía.
    make_gradcam_heatmap(escenario.modelo, escenario.tensor)

    # Assert — el subgrafo usa la capa requerida y recibe el tensor original.
    escenario.modelo.get_layer.assert_called_once_with("conv10_thisone")
    escenario.backend.keras.Model.assert_called_once_with(
        escenario.modelo.inputs, [escenario.capa.output, escenario.modelo.output]
    )
    escenario.submodelo.assert_called_once_with(escenario.tensor)


def test_gradcam_informa_capa_ausente_y_conserva_causa(gradcam_simulado):
    # Arrange — simular el error de búsqueda de una capa inexistente.
    escenario = gradcam_simulado
    causa = ValueError("capa ausente")
    escenario.modelo.get_layer.side_effect = causa

    # Act — intentar generar un mapa sin la capa requerida.
    with pytest.raises(ValueError, match="Grad-CAM necesita la capa 'conv10_thisone'") as error:
        make_gradcam_heatmap(escenario.modelo, escenario.tensor)

    # Assert — el diagnóstico conserva la causa y falla antes de crear el subgrafo.
    assert error.value.__cause__ is causa
    escenario.backend.keras.Model.assert_not_called()


def test_gradcam_selecciona_clase_ganadora_si_no_se_indica_indice(gradcam_simulado):
    # Arrange — la clase 1 es la ganadora en las predicciones programadas.
    escenario = gradcam_simulado

    # Act — omitir la clase objetivo para activar argmax.
    make_gradcam_heatmap(escenario.modelo, escenario.tensor)

    # Assert — la pérdida corresponde a la clase ganadora y a sus activaciones.
    escenario.cinta.gradient.assert_called_once()
    perdida, activaciones = escenario.cinta.gradient.call_args.args
    np.testing.assert_allclose(perdida, [0.7])
    assert activaciones is escenario.activaciones


def test_gradcam_respeta_clase_cero_explicita_aunque_no_sea_ganadora(gradcam_simulado):
    # Arrange — la clase 0 tiene probabilidad 0.1, inferior a la ganadora.
    escenario = gradcam_simulado

    # Act — solicitar explícitamente la clase cero.
    make_gradcam_heatmap(escenario.modelo, escenario.tensor, class_index=0)

    # Assert — el cero no se confunde con la ausencia de índice.
    escenario.cinta.gradient.assert_called_once()
    perdida = escenario.cinta.gradient.call_args.args[0]
    np.testing.assert_allclose(perdida, [0.1])


def test_gradcam_pondera_canales_aplica_relu_y_normaliza(gradcam_simulado):
    # Arrange — medias por canal [1, 2]; mapa ponderado [[-2, 3], [4, 6]].
    escenario = gradcam_simulado
    escenario.activaciones = _tensor([[[[-2, 0], [1, 1]], [[2, 1], [0, 3]]]])
    escenario.gradientes = _tensor([[[[0, 1], [2, 3]], [[0, 1], [2, 3]]]])
    esperado = np.array([[0, 0.5], [2 / 3, 1]], dtype=np.float32)

    # Act — ejecutar la aritmética real del módulo con operaciones NumPy simuladas.
    mapa = make_gradcam_heatmap(escenario.modelo, escenario.tensor)

    # Assert — valores correctos y contrato 2D (h, w) cumplido.
    np.testing.assert_allclose(mapa, esperado)
    assert mapa.dtype == np.float32
    assert mapa.shape == (2, 2)
    assert mapa.min() == 0.0
    assert mapa.max() == 1.0
    escenario.backend.reduce_mean.assert_called_once_with(escenario.gradientes, axis=(0, 1, 2))


def test_gradcam_devuelve_ceros_finitos_cuando_gradientes_son_nulos(gradcam_simulado):
    # Arrange — anular los gradientes manteniendo activaciones no nulas.
    escenario = gradcam_simulado
    escenario.gradientes = _tensor(np.zeros((1, 2, 2, 2)))

    # Act — generar el mapa de una contribución nula.
    mapa = make_gradcam_heatmap(escenario.modelo, escenario.tensor)

    # Assert — no hay división por cero ni valores no finitos.
    np.testing.assert_array_equal(mapa, np.zeros_like(mapa))
    assert np.isfinite(mapa).all()


def test_gradcam_anula_contribuciones_totalmente_negativas(gradcam_simulado):
    # Arrange — activaciones positivas y pesos negativos en todos los canales.
    escenario = gradcam_simulado
    escenario.gradientes = _tensor(-np.ones((1, 2, 2, 2)))

    # Act — aplicar el cálculo que incluye ReLU.
    mapa = make_gradcam_heatmap(escenario.modelo, escenario.tensor)

    # Assert — las contribuciones negativas no aparecen en el mapa.
    np.testing.assert_array_equal(mapa, np.zeros_like(mapa))


def test_superposicion_acepta_mapa_bidimensional_generado_por_gradcam(gradcam_simulado):
    # Arrange — el mapa de producción ya cumple el contrato 2D y puede componerse.
    escenario = gradcam_simulado
    mapa = make_gradcam_heatmap(escenario.modelo, escenario.tensor)
    imagen = np.zeros((4, 6, 3), dtype=np.uint8)

    # Act — componer directamente ambas funciones de producción.
    resultado = overlay_heatmap(mapa, imagen)

    # Assert — el mapa es 2D y la superposición ya no lanza error.
    assert mapa.shape == (2, 2)
    assert resultado.shape == imagen.shape
    assert resultado.dtype == np.uint8


def test_superposicion_rechaza_imagen_bidimensional():
    # Arrange — mapa válido e imagen gris sin el eje de canales BGR.
    mapa = np.zeros((2, 2), dtype=np.float32)
    imagen = np.zeros((4, 6), dtype=np.uint8)

    # Act — intentar superponer sobre una imagen bidimensional.
    with pytest.raises(ValueError, match="imagen debe ser BGR") as error:
        overlay_heatmap(mapa, imagen)

    # Assert — el diagnóstico incluye la forma incompatible.
    assert str(imagen.shape) in str(error.value)


def test_superposicion_rechaza_imagen_de_cuatro_canales():
    # Arrange — mapa válido e imagen con un canal de más.
    mapa = np.zeros((2, 2), dtype=np.float32)
    imagen = np.zeros((4, 6, 4), dtype=np.uint8)

    # Act — intentar superponer sobre una imagen de cuatro canales.
    with pytest.raises(ValueError, match="imagen debe ser BGR") as error:
        overlay_heatmap(mapa, imagen)

    # Assert — se identifica la forma rechazada.
    assert str(imagen.shape) in str(error.value)


def test_superposicion_rechaza_alpha_negativo():
    # Arrange — entradas válidas y transparencia por debajo del límite.
    mapa = np.zeros((2, 2), dtype=np.float32)
    imagen = np.zeros((4, 6, 3), dtype=np.uint8)

    # Act — solicitar una mezcla con alpha negativo.
    with pytest.raises(ValueError, match=r"alpha debe estar en \[0, 1\]") as error:
        overlay_heatmap(mapa, imagen, alpha=-0.1)

    # Assert — el diagnóstico identifica el valor recibido.
    assert "recibí -0.1" in str(error.value)


def test_superposicion_rechaza_alpha_mayor_que_uno():
    # Arrange — entradas válidas y transparencia por encima del límite.
    mapa = np.zeros((2, 2), dtype=np.float32)
    imagen = np.zeros((4, 6, 3), dtype=np.uint8)

    # Act — solicitar una mezcla con alpha mayor que uno.
    with pytest.raises(ValueError, match=r"alpha debe estar en \[0, 1\]") as error:
        overlay_heatmap(mapa, imagen, alpha=1.1)

    # Assert — el diagnóstico identifica el valor recibido.
    assert "recibí 1.1" in str(error.value)


def test_superposicion_redimensiona_y_mezcla_con_alpha_predeterminado():
    # Arrange — referencia OpenCV para un mapa menor que la imagen rectangular.
    mapa = np.array([[0, 1], [0.5, 0.25]], dtype=np.float32)
    imagen = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
    calor = cv2.resize(np.uint8(255 * mapa), (6, 4))
    colores = cv2.applyColorMap(calor, cv2.COLORMAP_JET)
    esperado = cv2.addWeighted(imagen, 0.6, colores, 0.4, 0)

    # Act — usar la transparencia predeterminada del módulo.
    resultado = overlay_heatmap(mapa, imagen)

    # Assert — la mezcla conserva dimensiones y tipo de la imagen BGR.
    np.testing.assert_array_equal(resultado, esperado)
    assert resultado.shape == imagen.shape
    assert resultado.dtype == np.uint8


def test_superposicion_con_alpha_cero_conserva_imagen_original():
    # Arrange — imagen no uniforme y mapa válido de otro tamaño.
    mapa = np.array([[0, 1], [0.5, 0.25]], dtype=np.float32)
    imagen = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
    esperado = imagen.copy()

    # Act — desactivar por completo la contribución del mapa.
    resultado = overlay_heatmap(mapa, imagen, alpha=0.0)

    # Assert — todos los píxeles conservan los valores originales.
    np.testing.assert_array_equal(resultado, esperado)
