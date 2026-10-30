"""Pruebas de la infraestructura de dobles — responsable: EL INICIADOR.

Este archivo no prueba el negocio: prueba que los dobles (FakeModel,
FakeNet) y las fixtures de data/ cumplen su contrato. Si estos nueve
tests están en verde, el equipo de los otros 111 puede confiar
plenamente en conftest.py.
"""

from pathlib import Path

import numpy as np
import pytest

from src.models.load_model import _validate_architecture


def test_fakemodel_predict_devuelve_distribucion_de_tres_clases(fake_model):
    # Arrange — el doble por defecto ya está inyectado

    # Act — pedir UNA predicción (el doble no lee la entrada)
    salida = fake_model.predict(np.zeros((1, 2, 2, 1)))

    # Assert — batch (1, 3) y suma ≈ 1: una distribución válida
    assert salida.shape == (1, 3)
    assert np.isclose(salida.sum(), 1.0)


def test_fakemodel_programado_en_bacteriana_argmax_ok(fakemodel_factory):
    # Arrange — programar la clase 0 (bacteriana)
    modelo = fakemodel_factory(clase_programada=0)

    # Act — una única predicción
    salida = modelo.predict(np.zeros((1, 2, 2, 1)))

    # Assert — el argmax señala exactamente la clase programada
    assert salida[0].argmax() == 0


def test_fakemodel_programado_en_normal_argmax_ok(fakemodel_factory):
    # Arrange — programar la clase 1 (normal)
    modelo = fakemodel_factory(clase_programada=1)

    # Act — una única predicción
    salida = modelo.predict(np.zeros((1, 2, 2, 1)))

    # Assert — el argmax señala exactamente la clase programada
    assert salida[0].argmax() == 1


def test_fakemodel_programado_en_viral_argmax_ok(fakemodel_factory):
    # Arrange — programar la clase 2 (viral)
    modelo = fakemodel_factory(clase_programada=2)

    # Act — una única predicción
    salida = modelo.predict(np.zeros((1, 2, 2, 1)))

    # Assert — el argmax señala exactamente la clase programada
    assert salida[0].argmax() == 2


def test_fakemodel_cuenta_los_llamados_al_predict(fake_model):
    # Arrange — el spy arranca en cero (fixture recién creada)

    # Act — invocar el predict dos veces
    fake_model.predict(np.zeros((1, 2, 2, 1)))
    fake_model.predict(np.zeros((1, 2, 2, 1)))

    # Assert — contó exactamente dos
    assert fake_model.llamadas == 2


def test_fakenet_pasa_la_validacion_del_wilhemnet(fake_net):
    # Arrange — arquitectura por defecto: conv10_thisone + 3 salidas

    # Act — validar como lo hace load_model antes de servir el modelo
    _validate_architecture(fake_net)

    # Assert — no lanzó error y expone la capa que exige Grad-CAM
    assert "conv10_thisone" in {capa.name for capa in fake_net.layers}


def test_fakenet_sin_capa_conv10_es_rechazado(fakenet_factory):
    # Arrange — simular un .h5 equivocado: sin la capa que exige Grad-CAM
    red = fakenet_factory(con_conv10=False)

    # Act + Assert — debe fallar NOMBRANDO la capa ausente
    with pytest.raises(ValueError, match="conv10_thisone"):
        _validate_architecture(red)


def test_fakenet_con_otra_salida_es_rechazado(fakenet_factory):
    # Arrange — simular una topología que entrega 2 clases en vez de 3
    red = fakenet_factory(clases_salida=2)

    # Act + Assert — debe fallar recordando el contrato de 3 clases
    with pytest.raises(ValueError, match="3 clases"):
        _validate_architecture(red)


def test_fixtures_apuntan_a_datos_reales(ruta_dcm, ruta_jpg):
    # Arrange — rutas entregadas por conftest.py

    # Act — preguntar si existen como archivos en el repo
    existe_dcm = Path(ruta_dcm).is_file()
    existe_jpg = Path(ruta_jpg).is_file()

    # Assert — data/ committeada sirve a todo el equipo
    assert existe_dcm
    assert existe_jpg
