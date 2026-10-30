"""Pruebas AAA del script de descarga del modelo: sin red, con un opener simulado."""

import hashlib
import io
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import download_model

CONTENIDO = b"pesos simulados del modelo"
SHA_CONTENIDO = hashlib.sha256(CONTENIDO).hexdigest()


def _opener(contenido: bytes = CONTENIDO) -> Mock:
    """Simula urlopen: devuelve un flujo de bytes usable como context manager."""
    respuesta = io.BytesIO(contenido)
    return Mock(return_value=respuesta)


def test_sha256_of_calcula_el_hash_del_archivo(tmp_path):
    # Arrange
    ruta = tmp_path / "archivo.bin"
    ruta.write_bytes(CONTENIDO)
    # Act
    resultado = download_model.sha256_of(ruta)
    # Assert
    assert resultado == SHA_CONTENIDO


def test_descarga_escribe_el_modelo_en_el_destino(tmp_path):
    # Arrange
    destino = tmp_path / "modelo.h5"
    # Act
    download_model.download_model(destino, "http://x/m.h5", SHA_CONTENIDO, opener=_opener())
    # Assert
    assert destino.read_bytes() == CONTENIDO


def test_descarga_usa_la_url_indicada(tmp_path):
    # Arrange
    opener = _opener()
    # Act
    download_model.download_model(
        tmp_path / "modelo.h5", "http://x/m.h5", SHA_CONTENIDO, opener=opener
    )
    # Assert
    assert opener.call_args.args[0] == "http://x/m.h5"


def test_no_descarga_si_el_modelo_valido_ya_existe(tmp_path):
    # Arrange
    destino = tmp_path / "modelo.h5"
    destino.write_bytes(CONTENIDO)
    opener = _opener()
    # Act
    download_model.download_model(destino, "http://x/m.h5", SHA_CONTENIDO, opener=opener)
    # Assert
    opener.assert_not_called()


def test_reemplaza_un_modelo_existente_corrupto(tmp_path):
    # Arrange
    destino = tmp_path / "modelo.h5"
    destino.write_bytes(b"archivo truncado")
    # Act
    download_model.download_model(destino, "http://x/m.h5", SHA_CONTENIDO, opener=_opener())
    # Assert
    assert destino.read_bytes() == CONTENIDO


def test_hash_distinto_lanza_error(tmp_path):
    # Arrange
    opener = _opener(b"contenido alterado")
    # Act / Assert
    with pytest.raises(ValueError, match="SHA-256"):
        download_model.download_model(
            tmp_path / "modelo.h5", "http://x/m.h5", SHA_CONTENIDO, opener=opener
        )


def test_hash_distinto_no_deja_el_modelo_en_el_destino(tmp_path):
    # Arrange
    destino = tmp_path / "modelo.h5"
    # Act
    with pytest.raises(ValueError):
        download_model.download_model(
            destino, "http://x/m.h5", SHA_CONTENIDO, opener=_opener(b"alterado")
        )
    # Assert
    assert not destino.exists()


def test_hash_distinto_no_deja_archivos_parciales(tmp_path):
    # Arrange
    destino = tmp_path / "modelo.h5"
    # Act
    with pytest.raises(ValueError):
        download_model.download_model(
            destino, "http://x/m.h5", SHA_CONTENIDO, opener=_opener(b"alterado")
        )
    # Assert
    assert list(tmp_path.iterdir()) == []


def test_destino_por_defecto_respeta_model_path(monkeypatch, tmp_path):
    # Arrange
    ruta = tmp_path / "otro.h5"
    monkeypatch.setenv("MODEL_PATH", str(ruta))
    # Act
    destino = download_model.default_destination()
    # Assert
    assert destino == ruta


def test_destino_por_defecto_sin_model_path_es_el_checkpoint_esperado(monkeypatch):
    # Arrange
    monkeypatch.delenv("MODEL_PATH", raising=False)
    # Act
    destino = download_model.default_destination()
    # Assert
    assert destino == Path("WilhemNet_86.h5")


def test_main_devuelve_cero_si_la_descarga_es_valida(monkeypatch, tmp_path):
    # Arrange
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "modelo.h5"))
    monkeypatch.setattr(download_model, "MODEL_SHA256", SHA_CONTENIDO)
    monkeypatch.setattr(download_model, "urlopen", _opener())
    # Act
    codigo = download_model.main()
    # Assert
    assert codigo == 0


def test_main_devuelve_uno_si_el_hash_no_coincide(monkeypatch, tmp_path):
    # Arrange
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "modelo.h5"))
    monkeypatch.setattr(download_model, "MODEL_SHA256", SHA_CONTENIDO)
    monkeypatch.setattr(download_model, "urlopen", _opener(b"alterado"))
    # Act
    codigo = download_model.main()
    # Assert
    assert codigo == 1
