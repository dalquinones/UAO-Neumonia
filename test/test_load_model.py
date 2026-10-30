"""Pruebas AAA de arquitectura y carga, sin TensorFlow ni checkpoints reales."""

import builtins
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from src.models.load_model import (
    CONV_LAYER_NAME,
    DEFAULT_MODEL_PATH,
    OUTPUT_CLASSES,
    _validate_architecture,
    load_model,
)


@pytest.fixture(autouse=True)
def tensorflow_simulado(monkeypatch, fake_net):
    """Sustituye TensorFlow por un módulo mínimo y entrega el cargador espía."""
    cargador = Mock(return_value=fake_net)
    modulo = ModuleType("tensorflow")
    modulo.keras = SimpleNamespace(models=SimpleNamespace(load_model=cargador))
    monkeypatch.setitem(sys.modules, "tensorflow", modulo)
    return cargador


def _vigilar_importaciones_tensorflow(monkeypatch):
    """Impide importar TensorFlow y registra cualquier intento, incluso del doble."""
    importar_original = builtins.__import__
    intentos = Mock()

    def importar(nombre, globals=None, locals=None, fromlist=(), level=0):
        if nombre == "tensorflow" or nombre.startswith("tensorflow."):
            intentos(nombre)
            raise AssertionError("No se debe intentar importar TensorFlow")
        return importar_original(nombre, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", importar)
    return intentos


def test_validacion_acepta_arquitectura_correcta(fake_net):
    # Arrange — la fixture contiene conv10_thisone y exactamente tres salidas.
    red = fake_net

    # Act — validar la arquitectura del doble.
    resultado = _validate_architecture(red)

    # Assert — la validación termina sin errores.
    assert resultado is None


def test_validacion_rechaza_modelo_sin_conv10_thisone(fakenet_factory):
    # Arrange — conservar tres clases, pero quitar la capa requerida.
    red = fakenet_factory(con_conv10=False)

    # Act — intentar validar la arquitectura incompleta.
    with pytest.raises(ValueError, match="conv10_thisone") as error:
        _validate_architecture(red)

    # Assert — el diagnóstico explica por qué se necesita la capa.
    assert "requerida por Grad-CAM" in str(error.value)


def test_validacion_rechaza_modelo_con_dos_clases(fakenet_factory):
    # Arrange — conservar la capa requerida y reducir las salidas.
    red = fakenet_factory(clases_salida=2)

    # Act — intentar validar un modelo con clases insuficientes.
    with pytest.raises(ValueError, match="debe producir 3 clases; produce 2") as error:
        _validate_architecture(red)

    # Assert — se informa de una arquitectura incompatible.
    assert "¿Es otro checkpoint?" in str(error.value)


def test_validacion_rechaza_modelo_con_cuatro_clases(fakenet_factory):
    # Arrange — conservar la capa requerida y aumentar las salidas.
    red = fakenet_factory(clases_salida=4)

    # Act — intentar validar un modelo con clases de más.
    with pytest.raises(ValueError, match="debe producir 3 clases; produce 4") as error:
        _validate_architecture(red)

    # Assert — se informa de una arquitectura incompatible.
    assert "¿Es otro checkpoint?" in str(error.value)


def test_ruta_inexistente_falla_antes_de_importar_tensorflow(monkeypatch):
    # Arrange — simular una ruta inexistente y vigilar los imports.
    ruta = Path("modelo_inexistente.h5")
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    intentos = _vigilar_importaciones_tensorflow(monkeypatch)

    # Act — solicitar un checkpoint que no existe.
    with pytest.raises(FileNotFoundError, match="No se encontró el archivo del modelo") as error:
        load_model(str(ruta))

    # Assert — se identifica la ruta sin intentar importar TensorFlow.
    assert str(ruta) in str(error.value)
    intentos.assert_not_called()


def test_importar_modulo_no_importa_tensorflow(monkeypatch):
    # Arrange — ejecutar el módulo en un espacio aislado, aunque ya esté importado.
    ruta = Path(__file__).resolve().parents[1] / "src" / "models" / "load_model.py"
    intentos = _vigilar_importaciones_tensorflow(monkeypatch)

    # Act — cargar las definiciones sin invocar la función de carga.
    espacio = runpy.run_path(str(ruta))

    # Assert — la función está disponible sin ningún intento de importar TensorFlow.
    assert callable(espacio["load_model"])
    intentos.assert_not_called()


def test_ruta_explicita_prevalece_sobre_variable_de_entorno(
    monkeypatch, tensorflow_simulado, fake_net
):
    # Arrange — las dos fuentes de ruta difieren y el archivo se simula existente.
    ruta = Path("modelo_explicito.h5")
    monkeypatch.setenv("MODEL_PATH", "modelo_entorno.h5")
    monkeypatch.setattr(Path, "is_file", lambda self: True)

    # Act — cargar usando una ruta explícita.
    modelo = load_model(str(ruta))

    # Assert — se carga solo la ruta explícita y se entrega el mismo modelo validado.
    tensorflow_simulado.assert_called_once_with(str(ruta), compile=False)
    assert modelo is fake_net


def test_sin_ruta_utiliza_model_path_del_entorno(monkeypatch, tensorflow_simulado):
    # Arrange — configurar la ruta del entorno y simular que existe el archivo.
    ruta = Path("modelo_entorno.h5")
    monkeypatch.setenv("MODEL_PATH", str(ruta))
    monkeypatch.setattr(Path, "is_file", lambda self: True)

    # Act — cargar sin indicar una ruta explícita.
    load_model()

    # Assert — se utiliza exactamente una vez la ruta del entorno.
    tensorflow_simulado.assert_called_once_with(str(ruta), compile=False)


def test_sin_ruta_ni_variable_utiliza_checkpoint_predeterminado(monkeypatch, tensorflow_simulado):
    # Arrange — eliminar la configuración del entorno y simular un archivo existente.
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.setattr(Path, "is_file", lambda self: True)

    # Act — cargar con la configuración predeterminada.
    load_model()

    # Assert — se solicita exactamente una vez el checkpoint esperado.
    tensorflow_simulado.assert_called_once_with(str(Path("WilhemNet_86.h5")), compile=False)


def test_carga_propaga_error_de_checkpoint_corrupto(monkeypatch, tensorflow_simulado):
    # Arrange — simular un archivo existente cuyo cargador falla al abrirlo.
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    causa = OSError("checkpoint corrupto")
    tensorflow_simulado.side_effect = causa

    # Act — intentar cargar el checkpoint corrupto.
    with pytest.raises(OSError, match="checkpoint corrupto") as error:
        load_model("corrupto.h5")

    # Assert — el error original se propaga sin ocultarlo ni sustituirlo.
    assert error.value is causa


def test_nombre_de_capa_gradcam_es_conv10_thisone():
    # Arrange / Act / Assert — Grad-CAM depende de este nombre exacto.
    assert CONV_LAYER_NAME == "conv10_thisone"


def test_el_modelo_debe_producir_tres_clases():
    # Arrange / Act / Assert — bacteriana, normal y viral.
    assert OUTPUT_CLASSES == 3


def test_ruta_predeterminada_es_el_checkpoint_wilhemnet():
    # Arrange / Act / Assert
    assert DEFAULT_MODEL_PATH == "WilhemNet_86.h5"


def test_carga_rechaza_checkpoint_sin_conv10_thisone(
    monkeypatch, tensorflow_simulado, fakenet_factory
):
    # Arrange — el archivo existe, pero el .h5 no expone la capa de Grad-CAM.
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    tensorflow_simulado.return_value = fakenet_factory(con_conv10=False)
    # Act / Assert
    with pytest.raises(ValueError, match="conv10_thisone"):
        load_model("otro.h5")


def test_carga_rechaza_checkpoint_con_dos_clases(monkeypatch, tensorflow_simulado, fakenet_factory):
    # Arrange
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    tensorflow_simulado.return_value = fakenet_factory(clases_salida=2)
    # Act / Assert
    with pytest.raises(ValueError, match="debe producir 3 clases; produce 2"):
        load_model("otro.h5")


def test_carga_valida_despues_de_abrir_el_checkpoint(
    monkeypatch, tensorflow_simulado, fakenet_factory
):
    # Arrange — la validación necesita el modelo ya cargado.
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    tensorflow_simulado.return_value = fakenet_factory(con_conv10=False)
    # Act
    with pytest.raises(ValueError):
        load_model("otro.h5")
    # Assert
    tensorflow_simulado.assert_called_once()


def test_carga_desactiva_la_compilacion(monkeypatch, tensorflow_simulado):
    # Arrange — Keras 3 rechaza la config de entrenamiento del checkpoint de 2022.
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    # Act
    load_model("modelo.h5")
    # Assert
    assert tensorflow_simulado.call_args.kwargs == {"compile": False}


def test_ruta_vacia_se_trata_como_no_indicada(monkeypatch, tensorflow_simulado):
    # Arrange — "" no es una ruta válida: se usa MODEL_PATH.
    monkeypatch.setenv("MODEL_PATH", "modelo_entorno.h5")
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    # Act
    load_model("")
    # Assert
    tensorflow_simulado.assert_called_once_with("modelo_entorno.h5", compile=False)


def test_model_path_inexistente_informa_la_ruta_del_entorno(monkeypatch, tmp_path):
    # Arrange
    ruta = tmp_path / "no_existe.h5"
    monkeypatch.setenv("MODEL_PATH", str(ruta))
    # Act
    with pytest.raises(FileNotFoundError) as error:
        load_model()
    # Assert
    assert str(ruta) in str(error.value)


def test_directorio_no_se_acepta_como_checkpoint(tmp_path, tensorflow_simulado):
    # Arrange — la ruta existe, pero no es un archivo.
    ruta = str(tmp_path)
    # Act
    with pytest.raises(FileNotFoundError, match="No se encontró el archivo del modelo"):
        load_model(ruta)
    # Assert
    tensorflow_simulado.assert_not_called()


def test_archivo_real_existente_se_entrega_al_cargador(tmp_path, tensorflow_simulado):
    # Arrange — archivo real en disco, sin simular la existencia.
    ruta = tmp_path / "modelo.h5"
    ruta.write_bytes(b"")
    # Act
    load_model(str(ruta))
    # Assert
    tensorflow_simulado.assert_called_once_with(str(ruta), compile=False)


def test_validacion_rechaza_modelo_sin_capas():
    # Arrange
    red = SimpleNamespace(layers=[], output_shape=(None, 3))
    # Act / Assert
    with pytest.raises(ValueError, match="conv10_thisone"):
        _validate_architecture(red)


def test_validacion_reporta_primero_la_capa_ausente(fakenet_factory):
    # Arrange — el modelo tiene los dos defectos a la vez.
    red = fakenet_factory(con_conv10=False, clases_salida=2)
    # Act / Assert — el diagnóstico prioriza la capa de Grad-CAM.
    with pytest.raises(ValueError, match="conv10_thisone"):
        _validate_architecture(red)


def test_validacion_acepta_conv10_en_cualquier_posicion():
    # Arrange — la capa requerida es la última de la red.
    capas = [SimpleNamespace(name="entrada"), SimpleNamespace(name="conv10_thisone")]
    red = SimpleNamespace(layers=capas, output_shape=(None, 3))
    # Act
    resultado = _validate_architecture(red)
    # Assert
    assert resultado is None
