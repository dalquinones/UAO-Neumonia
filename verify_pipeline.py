"""Verificador de recepción del entorno (equipo UAO-Neumonia).

Cada colaborador debe correrlo inmediatamente después de `uv sync`:

    uv run python verify_pipeline.py

Ejercita los 3 módulos ya implementados del refactor MVC usando las
imágenes de `data/` (que viajan en el repo) y un FakeModel (sin TF):

    1) read_img       → DICOM y JPG a (array BGR, preview PIL)
    2) preprocess_img → tensor (1, 512, 512, 1) float32 en [0, 1]
    3) integrator     → fachada con modelo inyectado

Si falla un check, el script aborta con código 1 (útil también en CI).
"""
import sys
from pathlib import Path

DATA = Path("data")


def check(nombre: str, condicion: bool, detalle: str = "") -> None:
    """Imprime una marca ✓/✗ y aborta el programa si falla."""
    marca = "✅" if condicion else "❌"
    print(f"{marca} {nombre}" + (f"  — {detalle}" if detalle else ""))
    if not condicion:
        sys.exit(1)


def main() -> None:
    # ── 1) read_img ──────────────────────────────────────────────
    from src.models.read_img import read_dicom, read_image

    array_dcm, preview_dcm = read_dicom(str(DATA / "normal.dcm"))
    check(
        "read_img · DICOM → 3 canales uint8",
        array_dcm.ndim == 3 and array_dcm.shape[2] == 3 and array_dcm.dtype.name == "uint8",
        f"shape={array_dcm.shape}",
    )
    check("read_img · preview DICOM es PIL.Image", type(preview_dcm).__name__ == "Image")

    array_jpg, _preview_jpg = read_image(str(DATA / "radiografia.jpg"))
    check(
        "read_img · JPG → 3 canales",
        array_jpg.ndim == 3 and array_jpg.shape[2] == 3,
        f"shape={array_jpg.shape}",
    )

    # ── 2) preprocess_img ────────────────────────────────────────
    from src.models.preprocess_img import preprocess

    tensor = preprocess(array_dcm)
    check(
        "preprocess · tensor (1, 512, 512, 1) float32",
        tensor.shape == (1, 512, 512, 1) and tensor.dtype.name == "float32",
    )
    check(
        "preprocess · valores en [0, 1]",
        0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0,
        f"rango {float(tensor.min()):.2f} - {float(tensor.max()):.2f}",
    )

    # ── 3) integrator (con modelo inyectado, sin TensorFlow) ────
    from src.controllers.integrator import LABELS, Integrator

    class FakeModel:
        """Stub con la misma interfaz que un modelo Keras."""
        def predict(self, x):
            assert x.shape == (1, 512, 512, 1), "contrato del tensor incumplido"
            return [[0.1, 0.7, 0.2]]

    resultado = Integrator(FakeModel()).analyze(str(DATA / "normal.dcm"))
    check(
        "integrator · label = clase del argmax",
        resultado["label"] == LABELS[1] == "normal",
        f'"{resultado["label"]}"',
    )
    check(
        "integrator · probabilidades como floats",
        resultado["probabilities"] == [0.1, 0.7, 0.2],
    )

    print("\n🎯 ENTORNO DE RECEPCIÓN OK — listo para escribir pruebas")


if __name__ == "__main__":
    main()
