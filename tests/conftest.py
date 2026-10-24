"""Fixtures compartidas por las pruebas."""

import os

import cv2
import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

RUTA_MODELO = "conv_MLP_84.h5"

hay_modelo = pytest.mark.skipif(
    not os.path.exists(RUTA_MODELO),
    reason="Requiere conv_MLP_84.h5 en la raiz del proyecto",
)


@pytest.fixture
def imagen_rgb():
    """Imagen RGB sintetica de 300x400 con valores deterministas."""
    generador = np.random.default_rng(42)
    return generador.integers(0, 256, (300, 400, 3), dtype=np.uint8)


@pytest.fixture
def archivo_dicom(tmp_path):
    """Crea un DICOM minimo valido y devuelve su ruta."""
    ruta = tmp_path / "prueba.dcm"
    pixeles = np.random.default_rng(7).integers(0, 4096, (64, 64)).astype(np.uint16)

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns = 64, 64
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = pixeles.tobytes()
    ds.save_as(ruta, enforce_file_format=True)
    return str(ruta)


@pytest.fixture
def archivo_jpg(tmp_path, imagen_rgb):
    """Guarda una imagen JPG temporal y devuelve su ruta."""
    ruta = tmp_path / "prueba.jpg"
    cv2.imwrite(str(ruta), imagen_rgb)
    return str(ruta)