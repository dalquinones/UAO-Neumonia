"""Fixtures compartidas del proyecto: pytest las inyecta por NOMBRE.

Los tests NUNCA importan tensorflow: el modelo entra inyectado como un
DOBLE de prueba con la misma interfaz que Keras (FakeModel.predict).
La red real (WilhemNet_86.h5) solo se ejercita en CI/Docker.

Uso del equipo — la fixture se pide por nombre en el parámetro del test:

    def test_ejemplo(integrator, ruta_dcm):
        ...

Infraestructura a cargo del INICIADOR del proyecto (no del code reviewer),
siguiendo el reparto de test/README.md.
"""

from pathlib import Path

import numpy as np
import pytest

from src.controllers.integrator import Integrator

RAIZ = Path(__file__).resolve().parents[1]  # raíz del proyecto
RUTA_DCM = RAIZ / "data" / "normal.dcm"
RUTA_JPG = RAIZ / "data" / "radiografia.jpg"


class _Capa:
    """Finge una capa de Keras: a la validación solo le interesa .name."""

    def __init__(self, nombre: str) -> None:
        self.name = nombre


class FakeModel:
    """Doble de la red Keras: .predict(tensor) devuelve probabilidades fijas.

    Con `clase_programada` se programa QUÉ clase "predice" (one-hot con
    margen 0.90/0.05 → argmax inequívoco) y `llamadas` funciona como SPY:
    test_integrator comprobará que el modelo se invoca EXACTAMENTE 1 vez.
    """

    def __init__(self, clase_programada: int = 0) -> None:
        self.clase_programada = clase_programada
        self.llamadas = 0
        self.ultima_entrada = None

    def predict(self, tensor) -> np.ndarray:
        """Interfaz Keras: recibe el batch (1,512,512,1) y devuelve (1,3)."""
        self.llamadas += 1
        self.ultima_entrada = tensor
        probabilidades = np.full(3, 0.05, dtype=np.float32)
        probabilidades[self.clase_programada] = 0.90
        return probabilidades[np.newaxis, :]


class FakeNet:
    """Doble de la ARQUITECTURA del .h5 (la topología, NO los pesos).

    Sirve a test_load_model: _validate_architecture() solo mira los
    nombres de las capas y la forma de salida — sin tocar TensorFlow.
    Por defecto es un WilhemNet_86 válido; con con_conv10=False (o
    clases_salida=2) simula un .h5 equivocado para probar el error.
    """

    def __init__(self, con_conv10: bool = True, clases_salida: int = 3) -> None:
        nombres = ["conv9_thisone", "dense_final"]
        if con_conv10:
            nombres.insert(0, "conv10_thisone")
        self.layers = [_Capa(n) for n in nombres]
        self.output_shape = (None, clases_salida)


# --------------------------- fixtures (por nombre) ---------------------------


@pytest.fixture
def fake_model() -> FakeModel:
    """FakeModel listo para inyectar (por defecto 'predice' bacteriana)."""
    return FakeModel()


@pytest.fixture
def fakemodel_factory() -> type[FakeModel]:
    """La CLASE FakeModel: para programar la respuesta según el caso."""
    return FakeModel


@pytest.fixture
def fake_net() -> FakeNet:
    """Arquitectura válida del WilhemNet_86, sin TensorFlow."""
    return FakeNet()


@pytest.fixture
def fakenet_factory() -> type[FakeNet]:
    """La CLASE FakeNet: para simular arquitecturas inválidas."""
    return FakeNet


@pytest.fixture
def integrator(fake_model) -> Integrator:
    """Controlador con el doble ya inyectado — la vista no importa TF."""
    return Integrator(model=fake_model)


@pytest.fixture
def ruta_dcm() -> str:
    """Radiografía DICOM real, committeada en data/."""
    return str(RUTA_DCM)


@pytest.fixture
def ruta_jpg() -> str:
    """Imagen JPG real, committeada en data/."""
    return str(RUTA_JPG)
