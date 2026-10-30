"""Dictamen PDF del resultado: datos del paciente, probabilidad más alta y dictamen.

Capa modelo pura (sin Streamlit ni TensorFlow). Usa fpdf2 con fuentes núcleo
(Latin-1), por eso los textos evitan caracteres fuera de ese juego.
"""

import re
from datetime import datetime

from fpdf import FPDF

APP_NAME = "Detector de Neumonía"

# Orden fijo del dataset (igual que LABELS del controlador).
_LABELS = ("bacteriana", "normal", "viral")

_HALLAZGOS = {
    "bacteriana": (
        "Hallazgos radiológicos compatibles con neumonía de probable origen bacteriano. "
        "Se sugiere correlación clínica, estudios de laboratorio y valoración médica "
        "para definir el tratamiento."
    ),
    "viral": (
        "Hallazgos radiológicos compatibles con neumonía de probable origen viral. "
        "Se sugiere correlación clínica, seguimiento de la evolución y valoración médica "
        "para definir el manejo."
    ),
    "normal": (
        "No se observan hallazgos radiológicos compatibles con neumonía. "
        "Correlacionar con la clínica del paciente."
    ),
}

_DISCLAIMER = (
    "Resultado generado por un modelo de inteligencia artificial como apoyo al diagnóstico. "
    "No reemplaza la valoración de un profesional de la salud."
)


def confidence_level(probability: float) -> str:
    """Nivel de confianza: alta (>= 80%), moderada (>= 60%) o baja (< 60%)."""
    if probability >= 0.80:
        return "alta"
    if probability >= 0.60:
        return "moderada"
    return "baja"


def build_verdict(label: str, probability: float) -> str:
    """Redacta el dictamen médico según la clase predicha y su probabilidad.

    Raises:
        ValueError: si la clase no es bacteriana, normal o viral.
    """
    if label not in _HALLAZGOS:
        raise ValueError(f"Clase desconocida: {label}")
    nivel = confidence_level(probability)
    dictamen = f"{_HALLAZGOS[label]} Predicción con confianza {nivel} ({probability * 100:.1f}%)."
    if nivel == "baja":
        dictamen += (
            " Dada la baja confianza, se recomienda confirmar con evaluación clínica "
            "y estudios complementarios."
        )
    return dictamen


def report_filename(patient_id: str, timestamp: datetime) -> str:
    """Nombre de archivo seguro: dictamen_<documento>_<AAAAMMDD_HHMMSS>.pdf."""
    documento = re.sub(r"[^A-Za-z0-9]", "", patient_id)
    return f"dictamen_{documento}_{timestamp:%Y%m%d_%H%M%S}.pdf"


def build_report_pdf(
    patient_name: str,
    patient_id: str,
    label: str,
    probabilities: list[float],
    timestamp: datetime,
) -> bytes:
    """Genera el PDF del dictamen y lo devuelve como bytes.

    Args:
        patient_name: nombre completo del paciente.
        patient_id: número de identificación.
        label: clase predicha (bacteriana, normal o viral).
        probabilities: probabilidades de las 3 clases; solo se reporta la más alta.
        timestamp: fecha y hora del resultado.

    Raises:
        ValueError: si falta el nombre o el documento, o la clase es desconocida.
    """
    nombre = patient_name.strip()
    documento = patient_id.strip()
    if not nombre or not documento:
        raise ValueError("El nombre y el número de identificación son obligatorios.")

    probabilidad = max(probabilities)
    dictamen = build_verdict(label, probabilidad)

    pdf = FPDF()
    # Sin compresión: el PDF es pequeño y su texto queda verificable en las pruebas.
    pdf.set_compression(False)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, APP_NAME, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, "Dictamen de radiografía de tórax", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.line(pdf.l_margin, pdf.get_y() + 3, pdf.w - pdf.r_margin, pdf.get_y() + 3)
    pdf.ln(10)

    _seccion(pdf, "Datos del paciente")
    _campo(pdf, "Nombre completo", nombre)
    _campo(pdf, "Identificación", documento)
    _campo(pdf, "Fecha y hora", f"{timestamp:%d/%m/%Y %H:%M:%S}")
    pdf.ln(4)

    _seccion(pdf, "Resultado")
    _campo(pdf, "Clase", label.capitalize())
    _campo(pdf, "Probabilidad", f"{probabilidad * 100:.1f}%")
    pdf.ln(4)

    _seccion(pdf, "Dictamen médico")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, dictamen, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, _DISCLAIMER, new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())


def _seccion(pdf: FPDF, titulo: str) -> None:
    """Pinta el título de una sección."""
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, titulo, new_x="LMARGIN", new_y="NEXT")


def _campo(pdf: FPDF, etiqueta: str, valor: str) -> None:
    """Pinta una fila «etiqueta: valor»."""
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(45, 7, f"{etiqueta}:")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, valor, new_x="LMARGIN", new_y="NEXT")
