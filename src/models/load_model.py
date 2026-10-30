"""Capa modelo: carga y validación del modelo WilhemNet_86.

Responsabilidad única: abrir el .h5 desde disco y GARANTIZAR que su
arquitectura es la que el resto del proyecto asume (la capa conv que
usan los mapas Grad-CAM, y salidas de 3 clases).

Este es el ÚNICO módulo del proyecto que conoce TensorFlow, y su
import es diferido (lazy): hasta que esta función se ejecute con un
archivo real, nada (tests, CI, Docker CPU) carga TF.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Solo lo ven los verificadores de tipos (ruff/mypy/ty): el runtime NUNCA
    # importa TF aquí — la carga real sigue diferida (lazy) dentro de la función.
    import tensorflow as tf

CONV_LAYER_NAME = "conv10_thisone"  # capa base de los mapas Grad-CAM
OUTPUT_CLASSES = 3  # bacteriana / normal / viral
DEFAULT_MODEL_PATH = "WilhemNet_86.h5"


def load_model(path: str | None = None) -> "tf.keras.Model":
    """Carga y valida el modelo entrenado.

    Args:
        path: ruta del .h5. Si no llega, usa MODEL_PATH del entorno o
              la ruta por defecto (raíz del proyecto).

    Returns:
        modelo de Keras listo para predict / Grad-CAM.

    Raises:
        FileNotFoundError: si el archivo del modelo no existe.
        ValueError: si la arquitectura no corresponde al WilhemNet_86.
    """
    ruta = Path(path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo del modelo: {ruta}")

    import tensorflow as tf  # lazy: solo si hay archivo real que abrir

    # compile=False: el checkpoint (2022, Keras 2) guarda `reduction='auto'` en la
    # loss, valor que Keras 3 (TF 2.16+) rechaza. Para inferencia solo se cargan
    # arquitectura + pesos; la config de entrenamiento no se necesita.
    modelo = tf.keras.models.load_model(str(ruta), compile=False)
    _validate_architecture(modelo)
    return modelo


def _validate_architecture(modelo: "tf.keras.Model") -> None:
    """Falla temprano si el .h5 no es el que esperamos."""
    nombres = {capa.name for capa in modelo.layers}
    if CONV_LAYER_NAME not in nombres:
        raise ValueError(
            f"El modelo no expone la capa '{CONV_LAYER_NAME}' (requerida por "
            f"Grad-CAM). ¿Cargaste el .h5 correcto?"
        )
    if modelo.output_shape[-1] != OUTPUT_CLASSES:
        raise ValueError(
            f"El modelo debe producir {OUTPUT_CLASSES} clases; produce "
            f"{modelo.output_shape[-1]}. ¿Es otro checkpoint?"
        )
