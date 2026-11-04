"""Única responsabilidad: descargar y cargar el clasificador entrenado una vez."""

from functools import lru_cache

# Modelo alternativo al .h5 original que el repositorio base no contiene.
MODEL_ID = "kiselyovd/chest-xray-classifier"
MODEL_REVISION = "09bd78b58c657870961faa1b79699da5ca8bc796"


@lru_cache(maxsize=1)
def cargar_modelo():
    """Devuelve el mismo modelo en llamadas sucesivas; requiere pesos o conexión."""
    from transformers import AutoModelForImageClassification

    modelo = AutoModelForImageClassification.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False, use_safetensors=True
    )
    return modelo.eval()
