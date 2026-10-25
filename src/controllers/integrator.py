"""Controlador: fachada única entre la vista y la capa modelo.

Regla MVC: la vista NUNCA importa tensorflow/cv2/pydicom. La vista solo
le pide cosas al Integrator, quien orquesta los módulos de src/models.

El modelo se INYECTA en el constructor (inyección de dependencias):
  - en producción : la red Keras real, cargada con load_model()
  - en pruebas    : un FakeModel sin TensorFlow
Así el controlador no sabe cuál está usando → se prueba localmente
sin TF y cambia de entorno (Mac/CI/Docker) sin tocar código.
"""
from pathlib import Path

import numpy as np

from src.models.preprocess_img import preprocess
from src.models.read_img import read_dicom, read_image

# Orden fijo del dataset de entrenamiento (3 carpetas por clase, alfabético)
LABELS = {0: "bacteriana", 1: "normal", 2: "viral"}


class Integrator:
    """Orquesta el pipeline completo: leer → preprocesar → predecir."""

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

    def _read_file(self, path: str):
        """Elige el lector correcto según la extensión del archivo."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"No se encontró el archivo: {path}")
        if p.suffix.lower() == ".dcm":
            return read_dicom(str(p))
        return read_image(str(p))