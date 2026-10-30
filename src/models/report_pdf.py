"""Informe PDF de apoyo diagnóstico: datos del paciente, estudio e impresión.

Capa modelo pura (sin Streamlit ni TensorFlow). Usa fpdf2 con fuentes núcleo
(Latin-1), por eso los textos evitan caracteres fuera de ese juego.

La redacción sigue la práctica de los informes radiológicos: estructura
estudio → resultado → impresión (ACR / RSNA RadReport) y léxico de certeza
estandarizado (Panicek y Hricak): "compatible con" (> 90 %), "sugestivo de"
(~75 %), "posible" (~50 %), "menos probable" (~25 %) e "improbable" (< 10 %).
"""

import re
from datetime import datetime

from fpdf import FPDF

APP_NAME = "Detector de Neumonía"

_NIVELES_NEUMONIA = {
    "compatible": "Hallazgos radiológicos compatibles con neumonía.",
    "sugestivo": "Hallazgos radiológicos sugestivos de neumonía.",
    "posible": "Posible neumonía.",
    "no_concluyente": "Hallazgos radiológicos no concluyentes para neumonía.",
}

_NIVELES_NORMAL = {
    "compatible": "Neumonía improbable: sin hallazgos radiológicos sugestivos de neumonía.",
    "sugestivo": "Neumonía menos probable: sin hallazgos radiológicos sugestivos de neumonía.",
    "posible": "Hallazgos radiológicos no concluyentes: no se puede descartar neumonía.",
    "no_concluyente": "Hallazgos radiológicos no concluyentes: no se puede descartar neumonía.",
}

# La radiografía sola no distingue de forma fiable la etiología bacteriana de la
# viral; el texto la presenta como asociación del modelo, no como diagnóstico.
_ETIOLOGIA = (
    "El patrón identificado por el modelo se asocia con mayor frecuencia a etiología "
    "{etiologia}; la radiografía por sí sola no permite confirmarla, por lo que debe "
    "establecerse con evaluación clínica y de laboratorio."
)

_RECOMENDACION = (
    "Dado el nivel de certeza, se recomienda valoración por médico radiólogo "
    "y estudios complementarios."
)

_FORMATOS = {".dcm": "DICOM", ".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}

_DISCLAIMER = (
    "Informe generado automáticamente por un modelo de inteligencia artificial con fines "
    "académicos; no es un dispositivo médico certificado ni constituye un diagnóstico. "
    "Requiere validación e interpretación por un médico radiólogo. El puntaje del modelo "
    "no está calibrado y no equivale a una probabilidad clínica."
)


def certainty_level(score: float) -> str:
    """Nivel del léxico de certeza: compatible, sugestivo, posible o no_concluyente."""
    if score >= 0.90:
        return "compatible"
    if score >= 0.75:
        return "sugestivo"
    if score >= 0.50:
        return "posible"
    return "no_concluyente"


def build_verdict(label: str, score: float) -> str:
    """Redacta la impresión diagnóstica según la clase predicha y su puntaje.

    Raises:
        ValueError: si la clase no es bacteriana, normal o viral.
    """
    if label not in ("bacteriana", "normal", "viral"):
        raise ValueError(f"Clase desconocida: {label}")
    nivel = certainty_level(score)
    if label == "normal":
        partes = [_NIVELES_NORMAL[nivel]]
    else:
        partes = [_NIVELES_NEUMONIA[nivel], _ETIOLOGIA.format(etiologia=label)]
    if nivel in ("posible", "no_concluyente"):
        partes.append(_RECOMENDACION)
    return " ".join(partes)


def report_filename(patient_id: str, timestamp: datetime) -> str:
    """Nombre de archivo seguro: informe_<documento>_<AAAAMMDD_HHMMSS>.pdf."""
    documento = re.sub(r"[^A-Za-z0-9]", "", patient_id)
    return f"informe_{documento}_{timestamp:%Y%m%d_%H%M%S}.pdf"


def build_report_pdf(
    patient_name: str,
    patient_id: str,
    label: str,
    probabilities: list[float],
    timestamp: datetime,
    file_extension: str,
) -> bytes:
    """Genera el PDF del informe y lo devuelve como bytes.

    Args:
        patient_name: nombre completo del paciente.
        patient_id: número de identificación.
        label: clase predicha (bacteriana, normal o viral).
        probabilities: puntajes de las 3 clases; solo se reporta el más alto.
        timestamp: fecha y hora del resultado.
        file_extension: extensión del archivo analizado (.dcm, .jpg, .jpeg, .png).

    Raises:
        ValueError: si falta el nombre o el documento, o la clase es desconocida.
    """
    nombre = patient_name.strip()
    documento = patient_id.strip()
    if not nombre or not documento:
        raise ValueError("El nombre y el número de identificación son obligatorios.")

    puntaje = max(probabilities)
    impresion = build_verdict(label, puntaje)
    extension = file_extension.lower()
    formato = _FORMATOS.get(extension, extension.lstrip(".").upper())

    pdf = FPDF()
    # Sin compresión: el PDF es pequeño y su texto queda verificable en las pruebas.
    pdf.set_compression(False)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, APP_NAME, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 6, "Informe de apoyo diagnóstico", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.line(pdf.l_margin, pdf.get_y() + 3, pdf.w - pdf.r_margin, pdf.get_y() + 3)
    pdf.ln(10)

    _seccion(pdf, "Datos del paciente")
    _campo(pdf, "Nombre completo", nombre)
    _campo(pdf, "Identificación", documento)
    _campo(pdf, "Fecha y hora", f"{timestamp:%d/%m/%Y %H:%M:%S}")
    pdf.ln(4)

    _seccion(pdf, "Estudio")
    _campo(pdf, "Tipo de estudio", "Radiografía de tórax")
    _campo(pdf, "Formato", formato)
    pdf.ln(4)

    _seccion(pdf, "Resultado del modelo")
    _campo(pdf, "Clase predicha", label.capitalize())
    _campo(pdf, "Puntaje del modelo", f"{puntaje * 100:.1f}%")
    pdf.ln(4)

    _seccion(pdf, "Impresión diagnóstica (apoyo por IA)")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, impresion, new_x="LMARGIN", new_y="NEXT")
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
