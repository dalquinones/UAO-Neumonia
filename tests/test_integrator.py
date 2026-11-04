"""Pruebas del controlador con motor inyectado, sin pesos ni internet."""

from io import BytesIO

import pytest
from PIL import Image

from src.neumonia.integrator import (
    Resultado,
    interpretar_probabilidades,
    nombre_legible,
    procesar_radiografia,
)


def test_interpreta_clase_predicha_y_tres_puntuaciones():
    # Arrange
    etiquetas = {0: "bacterial_pneumonia", 1: "normal", 2: "viral_pneumonia"}
    # Act
    resultado = interpretar_probabilidades([0.12, 0.08, 0.80], etiquetas)
    # Assert
    assert resultado.etiqueta == "Neumonía viral"
    assert resultado.confianza == pytest.approx(0.80)
    assert len(resultado.probabilidades) == 3


@pytest.mark.parametrize("valores", [[0.5, 0.6, 0.7], [-0.2, 0.3, 0.9], [float("nan"), 0.2, 0.8]])
def test_rechaza_vector_probabilidades_invalido(valores):
    with pytest.raises(ValueError):
        interpretar_probabilidades(valores, {0: "a", 1: "b", 2: "c"})


def test_etiqueta_desconocida_es_legible():
    assert nombre_legible("special_class") == "Special class"


def test_rechaza_indices_de_clase_incorrectos():
    with pytest.raises(ValueError, match="Índices"):
        interpretar_probabilidades([0.2, 0.8], {1: "normal", 2: "viral"})


def test_integrador_delega_en_motor_inyectado():
    # Arrange
    buffer = BytesIO()
    Image.new("RGB", (16, 16), color="gray").save(buffer, format="PNG")
    llamadas = []

    def motor_falso(imagen, generar_mapa=True):
        llamadas.append((imagen.size, generar_mapa))
        return Resultado("Normal", 0.95, {"Normal": 0.95, "Viral": 0.05})

    # Act
    imagen, resultado = procesar_radiografia(
        buffer.getvalue(), "radiografia.png", generar_mapa=False, motor=motor_falso
    )
    # Assert
    assert imagen.size == (16, 16)
    assert llamadas == [((16, 16), False)]
    assert resultado.etiqueta == "Normal"
