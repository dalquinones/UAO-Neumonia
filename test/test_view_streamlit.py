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


def test_vista_acepta_los_formatos_que_expone_la_capa_de_lectura():
    # Arrange / Act / Assert — sumar un formato (p. ej. NIfTI) solo toca read_img.
    from src.controllers.integrator import SUPPORTED_EXTENSIONS

    assert app.EXTENSIONES_PERMITIDAS == SUPPORTED_EXTENSIONS
    assert app.TIPOS_SUBIDA == [extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS]


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


def test_tipos_de_subida_no_llevan_punto():
    # Arrange / Act / Assert — st.file_uploader espera extensiones sin punto.
    assert all(not tipo.startswith(".") for tipo in app.TIPOS_SUBIDA)


def test_datos_paciente_rechaza_tabulaciones_y_saltos_de_linea():
    # Arrange / Act / Assert
    assert app.datos_paciente_completos("\t\n", "123") is False


def test_ahora_colombia_usa_la_zona_cot():
    # Arrange / Act
    fecha = app.ahora_colombia()
    # Assert
    assert fecha.tzname() == "COT"


def test_ahora_colombia_corresponde_al_instante_actual():
    # Arrange
    from datetime import UTC, datetime

    # Act
    diferencia = abs(app.ahora_colombia() - datetime.now(UTC))
    # Assert
    assert diferencia.total_seconds() < 5


def test_analizar_archivo_exitoso_no_incluye_error():
    # Arrange
    integrator = Mock()
    integrator.explain.return_value = {"overlay": Image.new("RGB", (2, 2))}
    # Act
    salida = app.analizar_archivo(integrator, "rx.png")
    # Assert
    assert "error" not in salida


def test_analizar_archivo_inexistente_usa_mensaje_amigable():
    # Arrange
    integrator = Mock()
    integrator.analyze.side_effect = FileNotFoundError("rx.png")
    # Act
    salida = app.analizar_archivo(integrator, "rx.png")
    # Assert
    assert salida["error"].startswith("No se encontró o no se pudo leer la imagen")


def test_analizar_archivo_normaliza_error_de_validacion_en_explain():
    # Arrange — el fallo ocurre en Grad-CAM, no en la predicción.
    integrator = Mock()
    integrator.explain.side_effect = ValueError("sin capa conv10_thisone")
    # Act
    salida = app.analizar_archivo(integrator, "rx.png")
    # Assert
    assert salida["error"].startswith("Error de validación")


def test_analizar_archivo_normaliza_error_inesperado_en_explain():
    # Arrange
    integrator = Mock()
    integrator.explain.side_effect = RuntimeError("fallo interno")
    # Act
    salida = app.analizar_archivo(integrator, "rx.png")
    # Assert
    assert salida["error"].startswith("Archivo corrupto o no soportado")


def test_guardar_subida_conserva_la_extension(tmp_path, monkeypatch):
    # Arrange
    from types import SimpleNamespace

    monkeypatch.setattr(app.tempfile, "tempdir", str(tmp_path))
    archivo = SimpleNamespace(name="rx.dcm", getvalue=lambda: b"DICM")
    # Act
    ruta = app._guardar_subida(archivo)
    # Assert
    assert ruta.endswith(".dcm")


def test_guardar_subida_escribe_el_contenido_subido(tmp_path, monkeypatch):
    # Arrange
    from pathlib import Path
    from types import SimpleNamespace

    monkeypatch.setattr(app.tempfile, "tempdir", str(tmp_path))
    archivo = SimpleNamespace(name="rx.png", getvalue=lambda: b"contenido")
    # Act
    ruta = app._guardar_subida(archivo)
    # Assert
    assert Path(ruta).read_bytes() == b"contenido"


def test_obtener_integrador_sin_modelo_devuelve_none(monkeypatch):
    # Arrange
    monkeypatch.setattr(app, "cargar_integrador", Mock(side_effect=FileNotFoundError("x")))
    monkeypatch.setattr(app.st, "error", Mock())
    # Act
    integrador = app.obtener_integrador()
    # Assert
    assert integrador is None


def test_obtener_integrador_sin_modelo_avisa_al_usuario(monkeypatch):
    # Arrange
    aviso = Mock()
    monkeypatch.setattr(app, "cargar_integrador", Mock(side_effect=FileNotFoundError("x")))
    monkeypatch.setattr(app.st, "error", aviso)
    # Act
    app.obtener_integrador()
    # Assert
    assert "WilhemNet_86.h5" in aviso.call_args.args[0]


def test_obtener_integrador_con_modelo_devuelve_el_controlador(monkeypatch):
    # Arrange
    controlador = Mock()
    monkeypatch.setattr(app, "cargar_integrador", Mock(return_value=controlador))
    # Act
    integrador = app.obtener_integrador()
    # Assert
    assert integrador is controlador


def test_app_smoke_sin_modelo_no_pide_datos_del_paciente(monkeypatch, tmp_path):
    # Arrange
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    ruta_app = Path(__file__).resolve().parents[1] / "src" / "views" / "app.py"
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "no_existe.h5"))
    at = AppTest.from_file(str(ruta_app), default_timeout=30)
    # Act
    at.run()
    # Assert — sin modelo la app se detiene antes del formulario.
    assert len(at.text_input) == 0
