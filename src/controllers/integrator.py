"""Controlador: fachada única entre la vista y la capa modelo.

Regla MVC: la vista NUNCA importa tensorflow/cv2/pydicom. La vista solo
le pide cosas al Integrator, quien orquesta los módulos de src/models.

El modelo se INYECTA en el constructor (inyección de dependencias):
  - en producción : la red Keras real, cargada con load_model()
  - en pruebas    : un FakeModel sin TensorFlow
Así el controlador no sabe cuál está usando → se prueba localmente
sin TF y cambia de entorno (Mac/CI/Docker) sin tocar código.
"""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.models.grad_cam import make_gradcam_heatmap, overlay_heatmap
from src.models.preprocess_img import preprocess
from src.models.read_img import read_dicom, read_image
from src.models.report_pdf import build_report_pdf, report_filename

__all__ = ["LABELS", "Integrator", "crear_integrador", "report_filename"]

# Orden fijo del dataset de entrenamiento (3 carpetas por clase, alfabético)
LABELS = {0: "bacteriana", 1: "normal", 2: "viral"}


class Integrator:
    """Orquesta el pipeline completo: leer → preprocesar → predecir → explicar."""

    def __init__(self, model):
        """Recibe un modelo con interfaz .predict(x) (Keras real o stub)."""
        self.model = model

    def analyze(self, path: str) -> dict:
        """Pipeline completo para una imagen o radiografía.

        Args:
            path: ruta del archivo (.dcm, .jpg, .jpeg, .png)

        Returns:
            dict con:
                label         -> clase predicha (str)
                probabilities -> list[float] con las 3 clases
                preview       -> PIL.Image para pintar en la vista

        Raises:
            FileNotFoundError: si el archivo no existe o no se lee.
        """
        array, preview = self._read_file(path)
        tensor = preprocess(array)
        probabilities = np.asarray(self.model.predict(tensor)).reshape(-1)
        idx = int(np.argmax(probabilities))
        return {
            "label": LABELS[idx],
            "probabilities": [float(p) for p in probabilities],
            "preview": preview,
        }

    def explain(self, path: str) -> dict:
        """Análisis completo con explicabilidad Grad-CAM para una imagen.

        Reutiliza lectura, preprocesamiento y predicción de :meth:`analyze` y
        añade el mapa de calor Grad-CAM compuesto sobre la imagen original,
        convertido a PIL RGB para que la vista lo pinte sin importar cv2.

        Args:
            path: ruta del archivo (.dcm, .jpg, .jpeg, .png)

        Returns:
            dict con:
                label         -> clase predicha (str)
                probabilities -> list[float] con las 3 clases
                preview       -> PIL.Image de la imagen original
                overlay       -> PIL.Image RGB con el mapa Grad-CAM superpuesto

        Raises:
            FileNotFoundError: si el archivo no existe o no se lee.
            ValueError: si el modelo no expone la capa conv de Grad-CAM.
        """
        array, preview = self._read_file(path)
        tensor = preprocess(array)
        probabilities = np.asarray(self.model.predict(tensor)).reshape(-1)
        idx = int(np.argmax(probabilities))
        heatmap = make_gradcam_heatmap(self.model, tensor, class_index=idx)
        overlay = overlay_heatmap(heatmap, array)
        overlay_rgb = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        return {
            "label": LABELS[idx],
            "probabilities": [float(p) for p in probabilities],
            "preview": preview,
            "overlay": overlay_rgb,
        }

    def generate_report(
        self,
        patient_name: str,
        patient_id: str,
        resultado: dict,
        timestamp: datetime,
        file_extension: str,
    ) -> bytes:
        """Genera el informe PDF a partir del resultado de :meth:`analyze`.

        Args:
            patient_name: nombre completo del paciente.
            patient_id: número de identificación.
            resultado: dict con 'label' y 'probabilities'.
            timestamp: fecha y hora del resultado.
            file_extension: extensión del archivo analizado (.dcm, .jpg, .jpeg, .png).

        Returns:
            bytes del PDF listo para descargar.

        Raises:
            ValueError: si falta el nombre o el documento.
        """
        return build_report_pdf(
            patient_name,
            patient_id,
            resultado["label"],
            resultado["probabilities"],
            timestamp,
            file_extension,
        )

    def _read_file(self, path: str):
        """Elige el lector correcto según la extensión del archivo."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"No se encontró el archivo: {path}")
        if p.suffix.lower() == ".dcm":
            return read_dicom(str(p))
        return read_image(str(p))


def crear_integrador(path: str | None = None) -> Integrator:
    """Fábrica de producción: carga el modelo real y lo inyecta en el controlador.

    La carga es diferida (TF-lazy): TensorFlow solo se importa si hay un
    checkpoint real que abrir. La vista usa esta fábrica para obtener el
    controlador sin conocer los módulos de la capa modelo.
    """
    from src.models.load_model import load_model  # lazy: misma convención

    return Integrator(load_model(path))
