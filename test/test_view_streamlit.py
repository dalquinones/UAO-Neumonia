"""Pruebas AAA de la vista Streamlit: lógica pura testeable + smoke test.

La vista es una cáscara fina del MVC: no importa tensorflow/cv2/pydicom y solo
habla con el controlador. Aquí se cubren las funciones puras (extensión y datos
del paciente) y el envoltorio de errores, más un smoke test con
streamlit.testing.v1.AppTest.
"""

from unittest.mock import Mock

from PIL import Image

from src.views import app


def test_extension_permitida_acepta_formatos_soportados():
    # Arrange / Act / Assert — mayúsculas incluidas (la validación no distingue).
    for nombre in ("rx.dcm", "rx.jpg", "rx.jpeg", "rx.png", "rx.DCM", "rx.JPEG"):
        assert app.extension_permitida(nombre) is True


def test_extension_permitida_rechaza_formatos_desconocidos():
    # Arrange / Act / Assert — CSV/PDF quedan fuera del MVP (S4).
    for nombre in ("rx.pdf", "rx.csv", "rx.txt", "rx"):
        assert app.extension_permitida(nombre) is False


def test_analizar_archivo_devuelve_resultado_y_overlay():
    # Arrange
    resultado = {"label": "normal", "probabilities": [0.05, 0.90, 0.05], "preview": Mock()}
    overlay = Image.new("RGB", (2, 2))
    integrator = Mock()
    integrator.analyze.return_value = resultado
    integrator.explain.return_value = {"overlay": overlay}
    # Act
    salida = app.analizar_archivo(integrator, "rx.jpg")
    # Assert
    assert salida == {"resultado": resultado, "overlay": overlay}
    integrator.analyze.assert_called_once_with("rx.jpg")
    integrator.explain.assert_called_once_with("rx.jpg")


def test_analizar_archivo_normaliza_archivo_inexistente():
    # Arrange
    integrator = Mock()
    integrator.analyze.side_effect = FileNotFoundError("no se pudo leer")
    # Act
    salida = app.analizar_archivo(integrator, "rx.jpg")
    # Assert
    assert "error" in salida
    assert "no se pudo leer" in salida["error"]
    integrator.explain.assert_not_called()


def test_analizar_archivo_normaliza_error_de_validacion():
    # Arrange
    integrator = Mock()
    integrator.analyze.side_effect = ValueError("El array de entrada está vacío")
    # Act
    salida = app.analizar_archivo(integrator, "rx.jpg")
    # Assert
    assert "error" in salida
    assert "vacío" in salida["error"]


def test_analizar_archivo_normaliza_dicom_corrupto():
    # Arrange — pydicom lanza su propia excepción, que la vista no puede importar.
    from pydicom.errors import InvalidDicomError

    integrator = Mock()
    integrator.analyze.side_effect = InvalidDicomError("DICM ausente")
    # Act
    salida = app.analizar_archivo(integrator, "rx.dcm")
    # Assert — se convierte en un aviso amigable en lugar de un traceback.
    assert "error" in salida
    assert "corrupto" in salida["error"]
    integrator.explain.assert_not_called()


def test_datos_paciente_completos_exige_nombre_y_documento():
    # Arrange / Act / Assert — espacios en blanco no cuentan como dato.
    assert app.datos_paciente_completos("Ana Pérez", "1020304050") is True
    assert app.datos_paciente_completos("", "1020304050") is False
    assert app.datos_paciente_completos("Ana Pérez", "   ") is False
    assert app.datos_paciente_completos(" ", "") is False


def test_ahora_colombia_usa_utc_menos_5():
    # Arrange
    from datetime import timedelta

    # Act
    fecha = app.ahora_colombia()
    # Assert
    assert fecha.utcoffset() == timedelta(hours=-5)


def test_app_smoke_sin_modelo_muestra_aviso(monkeypatch, tmp_path):
    # Arrange — forzar la ausencia del checkpoint sin depender del entorno.
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    ruta_app = Path(__file__).resolve().parents[1] / "src" / "views" / "app.py"
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "no_existe.h5"))
    at = AppTest.from_file(str(ruta_app), default_timeout=30)
    # Act
    at.run()
    # Assert — sin excepciones, título presente y aviso del modelo ausente.
    assert not at.exception
    assert at.title[0].value == "🫁 Detector de neumonía"
    assert any("WilhemNet_86.h5" in error.value for error in at.error)
