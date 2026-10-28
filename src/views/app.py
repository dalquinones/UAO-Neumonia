"""Vista Streamlit del detector de neumonía (cáscara fina del MVC).

Regla dura (CONSTITUTION.md §5): esta vista NUNCA importa tensorflow/cv2/pydicom.
Solo habla con la capa controlador (Integrator y su fábrica). La carga del modelo
es diferida: sin `WilhemNet_86.h5` en la raíz, la app avisa de forma amigable y
mantiene la interfaz viva. La persistencia CSV/PDF queda fuera del MVP.

Ejecutar:

    uv run streamlit run src/views/app.py
"""

import tempfile
from pathlib import Path

import streamlit as st

from src.controllers.integrator import LABELS, Integrator, crear_integrador

EXTENSIONES_PERMITIDAS = (".dcm", ".jpg", ".jpeg", ".png")


def extension_permitida(nombre: str) -> bool:
    """True si el nombre del archivo tiene una extensión soportada."""
    return Path(nombre).suffix.lower() in EXTENSIONES_PERMITIDAS


def formatear_probabilidades(probabilidades: list[float]) -> list[str]:
    """Convierte las probabilidades a textos legibles «clase: NN.N%»."""
    return [f"{LABELS[i]}: {probabilidades[i] * 100:.1f}%" for i in range(len(probabilidades))]


def analizar_archivo(integrator: Integrator, path: str) -> dict:
    """Ejecuta analyze + explain y normaliza los errores a un mensaje amigable.

    Args:
        integrator: controlador con el modelo inyectado.
        path: ruta del archivo (.dcm/.jpg/.jpeg/.png).

    Returns:
        dict con 'resultado' y 'overlay' en caso de éxito, o 'error' (str).
    """
    try:
        resultado = integrator.analyze(path)
        explicacion = integrator.explain(path)
    except FileNotFoundError as error:
        return {"error": f"No se encontró o no se pudo leer la imagen: {error}"}
    except ValueError as error:
        return {"error": f"Error de validación: {error}"}
    # Red de seguridad de la frontera UI: un DICOM corrupto lanza InvalidDicomError
    # (pydicom), que la vista no puede importar por la regla de "vista pura". Sin
    # esto, el usuario vería un traceback en lugar de un aviso amigable.
    except Exception as error:  # noqa: BLE001
        return {"error": f"Archivo corrupto o no soportado: {error}"}
    return {"resultado": resultado, "overlay": explicacion["overlay"]}


@st.cache_resource
def cargar_integrador() -> Integrator:
    """Carga el modelo una sola vez (TF-lazy). Lanza FileNotFoundError si falta."""
    return crear_integrador()


def obtener_integrador() -> Integrator | None:
    """Devuelve el controlador listo, o None si el checkpoint no está."""
    try:
        return cargar_integrador()
    except FileNotFoundError as error:
        st.error(f"No se encontró el modelo `WilhemNet_86.h5`: {error}")
        return None


def _guardar_subida(archivo) -> str:
    """Guarda el archivo subido en un temporal y devuelve su ruta."""
    sufijo = Path(archivo.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=sufijo) as temporal:
        temporal.write(archivo.getvalue())
        return temporal.name


_CSS_CORPORATIVO = """
<style>
    /* Fondo corporativo (cian claro) */
    [data-testid="stAppViewContainer"] {
        background: #E8FDFD;
    }
    [data-testid="stHeader"] {
        background: transparent;
    }

    /* Contenido al 60% del ancho y centrado */
    [data-testid="stAppViewContainer"] .block-container {
        max-width: 60%;
        margin-left: auto;
        margin-right: auto;
    }

    /* Título principal: azul corporativo con línea de acento */
    [data-testid="stAppViewContainer"] h1 {
        color: #123a5c;
        font-weight: 700;
        letter-spacing: -0.5px;
        border-bottom: 3px solid #1f4e79;
        padding-bottom: 0.5rem;
    }

    /* Subtítulos de sección */
    [data-testid="stAppViewContainer"] h3 {
        color: #1f4e79;
        font-weight: 600;
    }

    /* Botón primario con relieve y hover */
    .stButton > button[kind="primary"] {
        background-color: #1f4e79;
        border: none;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(31, 78, 121, 0.35);
        transition: box-shadow 0.15s ease, background-color 0.15s ease;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #2a5f8f;
        box-shadow: 0 4px 12px rgba(31, 78, 121, 0.45);
    }

    /* Tarjetas: contenedores con borde */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border: 1px solid #dde4ee;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(18, 58, 92, 0.06);
    }

    /* Métricas como tarjetas */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dde4ee;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        box-shadow: 0 2px 8px rgba(18, 58, 92, 0.05);
    }

    /* Zona de subida con borde de acento */
    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #1f4e79;
        border-radius: 12px;
        background-color: #f8fafc;
    }
</style>
"""


def _inyectar_estilos() -> None:
    """Aplica el estilo corporativo (colores, tarjetas y relieve)."""
    st.markdown(_CSS_CORPORATIVO, unsafe_allow_html=True)


def main() -> None:
    """Punto de entrada de la interfaz Streamlit."""
    st.set_page_config(page_title="Detector de neumonía", page_icon="🫁", layout="wide")
    _inyectar_estilos()
    st.title("🫁 Detector de neumonía")
    st.caption(
        "Clasificación de radiografías de tórax (bacteriana · normal · viral) "
        "con explicabilidad Grad-CAM."
    )

    integrator = obtener_integrador()
    if integrator is None:
        return

    archivo = st.file_uploader("Subí una radiografía", type=["dcm", "jpg", "jpeg", "png"])

    if archivo is None:
        st.info("Subí una imagen (`.dcm`, `.jpg`, `.jpeg` o `.png`) para comenzar.")
        return

    if not extension_permitida(archivo.name):
        st.error(f"Formato no soportado: `{archivo.name}`.")
        return

    if st.button("Analizar", type="primary"):
        ruta = _guardar_subida(archivo)
        try:
            with st.spinner("Analizando la radiografía…"):
                resultado = analizar_archivo(integrator, ruta)
        finally:
            Path(ruta).unlink(missing_ok=True)

        if "error" in resultado:
            st.error(resultado["error"])
            return

        with st.container(border=True):
            st.subheader("Análisis visual")
            columna_preview, columna_overlay = st.columns(2)
            with columna_preview:
                st.markdown("**Previsualización**")
                st.image(resultado["resultado"]["preview"])
            with columna_overlay:
                st.markdown("**Grad-CAM**")
                st.image(resultado["overlay"])

        with st.container(border=True):
            st.subheader("Resultado")
            st.metric("Clase", resultado["resultado"]["label"])
            for texto in formatear_probabilidades(resultado["resultado"]["probabilities"]):
                st.write(texto)


if __name__ == "__main__":
    main()
