"""Pruebas AAA del informe PDF (capa modelo, sin Streamlit ni TensorFlow)."""

import re
from datetime import datetime

import pytest

from src.models import report_pdf

FECHA = datetime(2026, 10, 29, 14, 30, 5)


def _texto_pdf(contenido: bytes) -> str:
    """Extrae el texto pintado (operadores Tj) del PDF sin comprimir.

    Une las líneas con espacios (los párrafos se cortan al ajustar al ancho) y
    quita el escape de los paréntesis propio de las cadenas PDF.
    """
    lineas = re.findall(r"\(((?:\\.|[^\\)])*)\) Tj", contenido.decode("latin-1"))
    return " ".join(re.sub(r"\\(.)", r"\1", linea) for linea in lineas)


def _informe(label="normal", probabilidades=(0.10, 0.85, 0.05), formato=".png") -> str:
    """Genera un informe de prueba y devuelve su texto."""
    contenido = report_pdf.build_report_pdf(
        "Ana María Pérez", "1020304050", label, list(probabilidades), FECHA, formato
    )
    return _texto_pdf(contenido)


def test_nivel_de_certeza_sigue_el_lexico_radiologico():
    # Arrange / Act / Assert — compatible >= 90%, sugestivo >= 75%, posible >= 50%.
    assert report_pdf.certainty_level(0.95) == "compatible"
    assert report_pdf.certainty_level(0.90) == "compatible"
    assert report_pdf.certainty_level(0.89) == "sugestivo"
    assert report_pdf.certainty_level(0.75) == "sugestivo"
    assert report_pdf.certainty_level(0.74) == "posible"
    assert report_pdf.certainty_level(0.50) == "posible"
    assert report_pdf.certainty_level(0.49) == "no_concluyente"


@pytest.mark.parametrize(
    ("score", "frase"),
    [
        (0.95, "Hallazgos radiológicos compatibles con neumonía."),
        (0.80, "Hallazgos radiológicos sugestivos de neumonía."),
        (0.60, "Posible neumonía."),
        (0.45, "Hallazgos radiológicos no concluyentes para neumonía."),
    ],
)
def test_impresion_de_neumonia_usa_el_lexico_de_certeza(score, frase):
    # Arrange / Act
    impresion = report_pdf.build_verdict("bacteriana", score)
    # Assert
    assert impresion.startswith(frase)


@pytest.mark.parametrize(
    ("score", "frase"),
    [
        (0.95, "Neumonía improbable"),
        (0.80, "Neumonía menos probable"),
        (0.60, "no se puede descartar neumonía"),
    ],
)
def test_impresion_normal_usa_el_lexico_de_certeza(score, frase):
    # Arrange / Act
    impresion = report_pdf.build_verdict("normal", score)
    # Assert
    assert frase in impresion


@pytest.mark.parametrize(("label", "etiologia"), [("bacteriana", "bacteriana"), ("viral", "viral")])
def test_impresion_no_afirma_la_etiologia(label, etiologia):
    # Arrange / Act
    impresion = report_pdf.build_verdict(label, 0.95)
    # Assert — la etiología se presenta como asociación del modelo, no como diagnóstico.
    assert f"se asocia con mayor frecuencia a etiología {etiologia}" in impresion
    assert "no permite confirmarla" in impresion
    assert "probable origen" not in impresion


def test_impresion_normal_no_menciona_etiologia():
    # Arrange / Act
    impresion = report_pdf.build_verdict("normal", 0.95)
    # Assert
    assert "etiología" not in impresion


@pytest.mark.parametrize("score", [0.60, 0.45])
def test_impresion_con_baja_certeza_recomienda_valoracion_por_radiologo(score):
    # Arrange / Act
    impresion = report_pdf.build_verdict("viral", score)
    # Assert
    assert "valoración por médico radiólogo" in impresion


def test_impresion_con_alta_certeza_no_agrega_recomendacion_extra():
    # Arrange / Act
    impresion = report_pdf.build_verdict("viral", 0.95)
    # Assert
    assert "valoración por médico radiólogo" not in impresion


def test_impresion_rechaza_clase_desconocida():
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="Clase desconocida"):
        report_pdf.build_verdict("covid", 0.9)


def test_pdf_incluye_encabezado_paciente_fecha_e_impresion():
    # Arrange / Act
    texto = _informe()
    # Assert
    assert "Detector de Neumonía" in texto
    assert "Informe de apoyo diagnóstico" in texto
    assert "Ana María Pérez" in texto
    assert "1020304050" in texto
    assert "29/10/2026 14:30:05" in texto
    assert "Impresión diagnóstica (apoyo por IA)" in texto
    assert "Neumonía menos probable" in texto


def test_pdf_ya_no_se_titula_dictamen_medico():
    # Arrange / Act
    texto = _informe()
    # Assert
    assert "Dictamen médico" not in texto
    assert "Probabilidad" not in texto


def test_pdf_muestra_solo_el_puntaje_mas_alto_como_puntaje_del_modelo():
    # Arrange / Act
    texto = _informe("bacteriana", (0.70, 0.20, 0.10))
    # Assert — las otras dos categorías no aparecen en el PDF.
    assert "Puntaje del modelo" in texto
    assert "70.0%" in texto
    assert "20.0%" not in texto
    assert "10.0%" not in texto


@pytest.mark.parametrize(
    ("formato", "esperado"),
    [(".dcm", "DICOM"), (".DCM", "DICOM"), (".jpg", "JPEG"), (".jpeg", "JPEG"), (".png", "PNG")],
)
def test_pdf_incluye_la_seccion_estudio_con_el_formato(formato, esperado):
    # Arrange / Act
    texto = _informe(formato=formato)
    # Assert
    assert "Radiografía de tórax" in texto
    assert esperado in texto


def test_pdf_aviso_aclara_uso_academico_y_validacion_por_radiologo():
    # Arrange / Act
    texto = _informe()
    # Assert
    assert "fines académicos" in texto
    assert "no es un dispositivo médico certificado" in texto
    assert "médico radiólogo" in texto
    assert "no equivale a una probabilidad clínica" in texto


@pytest.mark.parametrize(("nombre", "documento"), [("", "123"), ("Ana", "  "), (" ", "")])
def test_pdf_exige_nombre_y_documento(nombre, documento):
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="obligatorios"):
        report_pdf.build_report_pdf(nombre, documento, "normal", [0.1, 0.8, 0.1], FECHA, ".png")


def test_nombre_de_archivo_incluye_documento_y_fecha():
    # Arrange / Act
    nombre = report_pdf.report_filename("10.203/04 50", FECHA)
    # Assert — el documento se limpia para que sea un nombre de archivo seguro.
    assert nombre == "informe_102030450_20261029_143005.pdf"
