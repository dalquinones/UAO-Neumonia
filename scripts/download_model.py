"""Descarga el checkpoint WilhemNet_86.h5 y verifica su integridad (SHA-256).

Los pesos (~112 MB) no se versionan en Git. Este script los descarga a la ruta
que usa load_model (MODEL_PATH o WilhemNet_86.h5 en la raíz) y solo deja el
archivo en su lugar si el hash coincide: la descarga es HTTP plano, así que el
checksum es la garantía de que el archivo no llegó alterado ni truncado.

Ejecutar:

    uv run python -m scripts.download_model     # o: make model
"""

import hashlib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO
from urllib.request import urlopen

from src.models.load_model import DEFAULT_MODEL_PATH

MODEL_URL = "http://208.76.80.10/WilhemNet_86.h5"
MODEL_SHA256 = "d9548ff6fd2c29cdd15328ff95e1ad1fc8c7226e04e86c5890526bea2215e18e"
_CHUNK = 1024 * 1024  # 1 MiB por lectura


def sha256_of(path: Path) -> str:
    """Calcula el SHA-256 de un archivo leyéndolo por bloques."""
    digest = hashlib.sha256()
    with path.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(_CHUNK), b""):
            digest.update(bloque)
    return digest.hexdigest()


def default_destination() -> Path:
    """Ruta destino: MODEL_PATH del entorno o el checkpoint por defecto."""
    return Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))


def download_model(
    destination: Path,
    url: str,
    expected_sha256: str,
    opener: Callable[[str], BinaryIO] = urlopen,
) -> Path:
    """Descarga el modelo si falta o está corrupto y verifica su SHA-256.

    Args:
        destination: ruta final del checkpoint.
        url: dirección de descarga.
        expected_sha256: hash esperado del archivo.
        opener: función que abre la URL (inyectable para pruebas sin red).

    Returns:
        la ruta del checkpoint verificado.

    Raises:
        ValueError: si el hash descargado no coincide (no se deja ningún archivo).
    """
    if destination.is_file() and sha256_of(destination) == expected_sha256:
        return destination

    parcial = destination.with_name(destination.name + ".part")
    try:
        with opener(url) as respuesta, parcial.open("wb") as salida:
            for bloque in iter(lambda: respuesta.read(_CHUNK), b""):
                salida.write(bloque)
        obtenido = sha256_of(parcial)
        if obtenido != expected_sha256:
            raise ValueError(
                f"El SHA-256 descargado ({obtenido}) no coincide con el esperado "
                f"({expected_sha256}). Archivo descartado."
            )
        parcial.replace(destination)
    finally:
        parcial.unlink(missing_ok=True)
    return destination


def main() -> int:
    """Punto de entrada de consola: 0 si el modelo quedó verificado, 1 si falló."""
    destino = default_destination()
    print(f"Obteniendo el modelo en {destino} (se descarga solo si falta o está corrupto)…")
    try:
        download_model(destino, MODEL_URL, MODEL_SHA256, opener=urlopen)
    except (OSError, ValueError) as error:
        print(f"❌ No se pudo obtener el modelo: {error}", file=sys.stderr)
        return 1
    print(f"✅ Modelo verificado (SHA-256 {MODEL_SHA256[:12]}…) en {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
