"""Pruebas AAA del dictamen PDF (capa modelo, sin Streamlit ni TensorFlow)."""

from datetime import datetime

import pytest

from src.models import report_pdf

FECHA = datetime(2026, 10, 29, 14, 30, 5)


def _texto_pdf(contenido: bytes) -> str:
    """Decodifica el PDF sin comprimir para buscar el texto pintado."""
    return contenido.decode("latin-1")


def test_nivel_de_confianza_por_umbral():
    # Arrange / Act / Assert — umbrales: alta >= 80%, moderada >= 60%, baja < 60%.
    assert report_pdf.confidence_level(0.80) == "alta"
    assert report_pdf.confidence_level(0.95) == "alta"
    assert report_pdf.confidence_level(0.60) == "moderada"
    assert report_pdf.confidence_level(0.79) == "moderada"
    assert report_pdf.confidence_level(0.59) == "baja"


@pytest.mark.parametrize(
    ("label", "fragmento"),
    [
        ("bacteriana", "origen bacteriano"),
        ("viral", "origen viral"),
        ("normal", "No se observan hallazgos"),
    ],
)
def test_dictamen_describe_la_clase_predicha(label, fragmento):
    # Arrange / Act
    dictamen = report_pdf.build_verdict(label, 0.92)
    # Assert
    assert fragmento in dictamen
    assert "confianza alta" in dictamen


def test_dictamen_con_confianza_baja_recomienda_confirmar():
    # Arrange / Act
    dictamen = report_pdf.build_verdict("viral", 0.45)
    # Assert
    assert "confianza baja" in dictamen
    assert "confirmar" in dictamen


def test_dictamen_rechaza_clase_desconocida():
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="Clase desconocida"):
        report_pdf.build_verdict("covid", 0.9)


def test_pdf_incluye_encabezado_paciente_fecha_probabilidad_y_dictamen():
    # Arrange
    probabilidades = [0.10, 0.85, 0.05]
    # Act
    contenido = report_pdf.build_report_pdf(
        "Ana María Pérez", "1020304050", "normal", probabilidades, FECHA
    )
    # Assert
    texto = _texto_pdf(contenido)
    assert contenido.startswith(b"%PDF")
    assert "Detector de Neumonía" in texto
    assert "Ana María Pérez" in texto
    assert "1020304050" in texto
    assert "29/10/2026 14:30:05" in texto
    assert "85.0%" in texto
    assert "No se observan hallazgos" in texto


def test_pdf_solo_muestra_la_probabilidad_mas_alta():
    # Arrange
    probabilidades = [0.70, 0.20, 0.10]
    # Act
    texto = _texto_pdf(
        report_pdf.build_report_pdf("Juan Gómez", "123", "bacteriana", probabilidades, FECHA)
    )
    # Assert — las otras dos categorías no aparecen en el PDF.
    assert "70.0%" in texto
    assert "20.0%" not in texto
    assert "10.0%" not in texto


@pytest.mark.parametrize(("nombre", "documento"), [("", "123"), ("Ana", "  "), (" ", "")])
def test_pdf_exige_nombre_y_documento(nombre, documento):
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="obligatorios"):
        report_pdf.build_report_pdf(nombre, documento, "normal", [0.1, 0.8, 0.1], FECHA)


def test_nombre_de_archivo_incluye_documento_y_fecha():
    # Arrange / Act
    nombre = report_pdf.report_filename("10.203/04 50", FECHA)
    # Assert — el documento se limpia para que sea un nombre de archivo seguro.
    assert nombre == "dictamen_102030450_20261029_143005.pdf"
