"""Vista Streamlit: muestra resultados; no conoce las librerías de inferencia."""

import streamlit as st

from .integrator import analizar_imagen, cargar_para_visualizacion


def main() -> None:
    """Construye la interfaz MVC sin contener reglas de predicción."""
    st.set_page_config(page_title="Proyecto Neumonía", page_icon="🫁", layout="wide")
    st.title("🫁 Detección experimental de neumonía")
    st.caption("Proyecto educativo basado en UAO-Neumonia")
    st.warning(
        "Uso académico exclusivamente. La predicción puede equivocarse: "
        "no reemplaza la evaluación de profesionales de salud."
    )
    archivo = st.file_uploader(
        "Seleccionar radiografía DICOM, JPEG o PNG", type=["dcm", "jpg", "jpeg", "png"]
    )
    crear_mapa = st.checkbox("Generar mapa Grad-CAM", value=True)
    if archivo is None:
        return
    try:
        imagen = cargar_para_visualizacion(archivo.getvalue(), archivo.name)
        izquierda, derecha = st.columns(2)
        izquierda.image(imagen, caption="Radiografía cargada", use_container_width=True)
        if st.button("Predecir", type="primary"):
            with st.spinner("Analizando radiografía (primera descarga puede requerir internet)..."):
                resultado = analizar_imagen(imagen, generar_mapa=crear_mapa)
            st.subheader("Resultado del clasificador")
            st.metric("Categoría prevista", resultado.etiqueta)
            st.metric("Puntuación softmax", f"{resultado.confianza:.1%}")
            st.caption("Puntuaciones del modelo no calibradas; no son probabilidades clínicas.")
            for nombre, puntuacion in resultado.probabilidades.items():
                st.write(f"{nombre}: {puntuacion:.1%}")
                st.progress(float(puntuacion))
            if resultado.heatmap is not None:
                derecha.image(
                    resultado.heatmap,
                    caption="Grad-CAM: influencia del modelo",
                    use_container_width=True,
                )
            elif crear_mapa:
                derecha.info("Se obtuvo la clasificación, pero no pudo generarse Grad-CAM.")
    except ValueError as exc:
        st.error(f"Imagen no válida: {exc}")
    except Exception as exc:
        st.error("La inferencia no se completó. Comprueba internet, memoria y dependencias.")
        st.code(f"{type(exc).__name__}: {exc}")
        st.info("No se generan predicciones simuladas si el modelo falla.")
    st.divider()
    st.caption("Modelo entrenado alternativo documentado en el README.")
