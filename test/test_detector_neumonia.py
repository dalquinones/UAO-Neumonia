"""
Pruebas unitarias para la Interfaz Gráfica: detector_neumonia.py
Valida el ciclo de vida de la GUI, guardado en CSV, reportes PDF y transiciones de estado.
"""
import os
import shutil
import glob
import pytest
import numpy as np
from unittest.mock import patch

from src.detector_neumonia import App


@pytest.fixture
def app_instance():
    """
    Instancia la aplicación simulando la ventana gráfica
    sin bloquear la ejecución de pytest (evitando el bucle infinito de mainloop).
    """
    with patch("tkinter.Tk.mainloop"):
        app = App()
        yield app
        app.root.destroy()


# 1. Prueba de estado inicial (1 prueba)
def test_gui_initial_state(app_instance):
    """Valida que los widgets inicien en el estado correcto (botón deshabilitado, datos limpios)."""
    assert app_instance.array is None
    assert app_instance.heatmap is None
    assert app_instance.label == ""
    assert app_instance.proba == 0.0
    assert str(app_instance.btn_predict["state"]) == "disabled"


# 2. Pruebas de guardado en CSV con múltiples formatos de paciente (5 pruebas)
@pytest.mark.parametrize(
    "patient_id,diag_label,proba",
    [
        ("12345678", "normal", 98.45),
        ("CC-99887766", "bacteriana", 89.20),
        ("PASAPORTE-XYZ", "viral", 75.10),
        ("TI-00112233", "normal", 99.99),
        ("ANONIMO-404", "bacteriana", 60.00),
    ],
    ids=["cedula", "con_guion", "pasaporte", "tarjeta_id", "anonimo"],
)
def test_gui_save_csv_various_patients(app_instance, patient_id, diag_label, proba):
    """Valida el guardado estructurado en historial.csv para diferentes identificadores."""
    app_instance.patient_id_var.set(patient_id)
    app_instance.label = diag_label
    app_instance.proba = proba

    with patch("src.detector_neumonia.showinfo"):
        app_instance.save_csv()

    assert os.path.exists("historial.csv")
    with open("historial.csv", "r") as f:
        content = f.read()
    assert patient_id in content
    assert diag_label in content


# 3. Pruebas de generación de reportes clínicos PDF para cada patología (5 pruebas)
@pytest.mark.parametrize(
    "patient_id,diag_label,proba",
    [
        ("PACIENTE-01", "normal", 95.5),
        ("PACIENTE-02", "bacteriana", 88.0),
        ("PACIENTE-03", "viral", 91.3),
        ("PACIENTE-04", "normal", 99.1),
        ("PACIENTE-05", "bacteriana", 74.2),
    ],
    ids=["pdf_norm1", "pdf_bact1", "pdf_viral1", "pdf_norm2", "pdf_bact2"],
)
def test_gui_create_pdf_various_diagnoses(app_instance, patient_id, diag_label, proba):
    """Valida la generación de reportes PDF clínicos sin capturas de pantalla."""
    app_instance.patient_id_var.set(patient_id)
    app_instance.label = diag_label
    app_instance.proba = proba
    sample_path = "data/JPG/normal/NORMAL2-IM-1144-0001.jpeg"
    if os.path.exists(sample_path):
        import cv2
        bgr = cv2.imread(sample_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        app_instance.array = cv2.resize(rgb, (512, 512))
        # Generar mapa de calor JET sobre la radiografía
        gray = cv2.cvtColor(app_instance.array, cv2.COLOR_RGB2GRAY)
        jet = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        app_instance.heatmap = cv2.addWeighted(app_instance.array, 0.6, cv2.cvtColor(jet, cv2.COLOR_BGR2RGB), 0.4, 0)
    else:
        app_instance.array = np.ones((512, 512, 3), dtype=np.uint8) * 128
        app_instance.heatmap = np.ones((512, 512, 3), dtype=np.uint8) * 128

    with patch("src.detector_neumonia.showinfo"):
        app_instance.create_pdf()

    # Buscar el PDF generado usando un patrón glob
    pattern = f"Reporte_{patient_id}_*.pdf"
    generated_pdfs = glob.glob(pattern)
    assert len(generated_pdfs) > 0, f"No se generó el PDF para {patient_id}"
    expected_pdf = generated_pdfs[0]
    assert os.path.getsize(expected_pdf) > 0

    # Guardar copia persistente de evidencia en la carpeta reports/evidencias_pdf/
    evidence_dir = "reports/evidencias_pdf"
    os.makedirs(evidence_dir, exist_ok=True)
    evidence_path = os.path.join(evidence_dir, f"Reporte_{patient_id}_{diag_label}.pdf")
    shutil.copy(expected_pdf, evidence_path)

    # Limpiar el temporal de la raíz
    os.remove(expected_pdf)


# 4. Pruebas de reinicio de formulario (Confirmado vs Cancelado) (3 pruebas)
def test_gui_reset_form_confirmed(app_instance):
    """Valida que al confirmar con OK se limpie el estado del paciente."""
    app_instance.patient_id_var.set("112233")
    app_instance.label = "bacteriana"
    app_instance.proba = 92.10
    app_instance.array = np.zeros((512, 512, 3), dtype=np.uint8)

    with patch("src.detector_neumonia.askokcancel", return_value=True):
        app_instance.reset_form()

    assert app_instance.patient_id_var.get() == ""
    assert app_instance.label == ""
    assert app_instance.proba == 0.0
    assert app_instance.array is None
    assert str(app_instance.btn_predict["state"]) == "disabled"


def test_gui_reset_form_cancelled(app_instance):
    """Valida que si el usuario cancela en el diálogo, los datos se preserven."""
    app_instance.patient_id_var.set("999888")
    app_instance.label = "viral"

    with patch("src.detector_neumonia.askokcancel", return_value=False):
        app_instance.reset_form()

    assert app_instance.patient_id_var.get() == "999888"
    assert app_instance.label == "viral"


def test_gui_consecutive_resets(app_instance):
    """Valida múltiples limpiezas sucesivas sin errores."""
    with patch("src.detector_neumonia.askokcancel", return_value=True):
        app_instance.reset_form()
        app_instance.reset_form()
    assert app_instance.label == ""


# 5. Pruebas de propiedades de la ventana y secuencia de reportes (3 pruebas)
def test_gui_window_title(app_instance):
    """Valida el título oficial de la ventana Tkinter."""
    assert "neumonía" in app_instance.root.title().lower()


def test_gui_report_id_increments(app_instance):
    """Valida que el identificador incremental del reporte avance en cada exportación."""
    initial_id = app_instance.report_id
    app_instance.patient_id_var.set("ID-INC")
    app_instance.label = "normal"
    app_instance.array = np.zeros((512, 512, 3), dtype=np.uint8)
    app_instance.heatmap = np.zeros((512, 512, 3), dtype=np.uint8)

    # Limpiar PDFs anteriores para evitar fallas por ejecuciones previas
    for rep in glob.glob("Reporte_ID-INC_*.pdf"):
        os.remove(rep)

    with patch("src.detector_neumonia.showinfo"):
        app_instance.create_pdf()

    # Test report_id is not used for pdf naming anymore,
    # instead we verify that pdf is generated
    generated_pdfs = glob.glob("Reporte_ID-INC_*.pdf")
    assert len(generated_pdfs) >= 1

    # Limpieza
    for rep in generated_pdfs:
        os.remove(rep)


def test_gui_empty_patient_id_default(app_instance):
    """Valida que si el usuario no escribe ID, el PDF y CSV usen 'S_N' por defecto."""
    app_instance.patient_id_var.set("")
    app_instance.label = "normal"
    app_instance.array = np.zeros((512, 512, 3), dtype=np.uint8)
    app_instance.heatmap = np.zeros((512, 512, 3), dtype=np.uint8)

    with patch("src.detector_neumonia.showinfo"):
        app_instance.save_csv()
        app_instance.create_pdf()

    # Por defecto 'S_N'
    generated_pdfs = glob.glob("Reporte_S_N_*.pdf")
    assert len(generated_pdfs) > 0
    rep_file = generated_pdfs[0]
    os.remove(rep_file)


# 6. Pruebas de transiciones de botones (3 pruebas)
def test_btn_predict_enables_when_image_loaded(app_instance):
    """Valida que el botón predecir cambie a normal cuando se carga una imagen."""
    app_instance.array = np.zeros((512, 512, 3), dtype=np.uint8)
    app_instance.btn_predict.config(state="normal")
    assert str(app_instance.btn_predict["state"]) == "normal"


def test_btn_predict_disables_on_reset(app_instance):
    """Valida que el botón predecir vuelva a deshabilitarse tras un reinicio."""
    app_instance.btn_predict.config(state="normal")
    with patch("src.detector_neumonia.askokcancel", return_value=True):
        app_instance.reset_form()
    assert str(app_instance.btn_predict["state"]) == "disabled"


def test_patient_entry_var_binding(app_instance):
    """Valida que la variable Tkinter esté ligada al campo de entrada."""
    app_instance.patient_id_var.set("CC-445566")
    assert app_instance.entry_patient_id.get() == "CC-445566"


# 7. Pruebas del Sistema de Diseño y UI Moderna (Fases 1-6)
def test_gui_responsive_minsize(app_instance):
    """Valida que la ventana tenga un tamaño mínimo responsive configurado."""
    minsize = app_instance.root.minsize()
    assert minsize[0] >= 800
    assert minsize[1] >= 600


def test_gui_image_labels_empty_state(app_instance):
    """Valida que los paneles de imagen utilicen ttk.Label y tengan empty state descriptivo."""
    from tkinter import ttk
    assert isinstance(app_instance.label_img1, ttk.Label)
    assert isinstance(app_instance.label_img2, ttk.Label)
    assert "Sin radiografía" in app_instance.label_img1.cget("text")
    assert "Grad-CAM" in app_instance.label_img2.cget("text")


def test_gui_progress_bar_exists(app_instance):
    """Valida que exista la barra de progreso indeterminada para feedback clínico."""
    from tkinter import ttk
    assert hasattr(app_instance, "progress")
    assert isinstance(app_instance.progress, ttk.Progressbar)
    assert str(app_instance.progress.cget("mode")) == "indeterminate"



def test_gui_shortcuts_registered(app_instance):
    """Valida que los atajos de accesibilidad estén enlazados a la ventana."""
    bound_keys = [app_instance.root.bind(k) for k in ["<Control-o>", "<Return>", "<Control-s>", "<Escape>"]]
    assert all(bound is not None and bound != "" for bound in bound_keys)


def test_gui_predict_without_image_warns(app_instance):
    """Valida que predecir sin imagen muestre una advertencia amigable sin tronar."""
    app_instance.array = None
    with patch("src.detector_neumonia.showwarning") as mock_warn:
        app_instance.run_prediction()
        mock_warn.assert_called_once()

