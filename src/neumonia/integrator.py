"""Controlador/orquestador: lectura → preprocesamiento → inferencia → Grad-CAM.

No importa la interfaz ni administra archivos en rutas absolutas.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .grad_cam import grad_cam, superponer_mapa
from .load_model import cargar_modelo
from .preprocess_img import preprocess_image
from .read_img import cargar_imagen

ETIQUETAS = {
    "bacterial_pneumonia": "Neumonía bacteriana",
    "bacterial": "Neumonía bacteriana",
    "normal": "Normal (sin neumonía)",
    "viral_pneumonia": "Neumonía viral",
    "viral": "Neumonía viral",
}


@dataclass(frozen=True)
class Resultado:
    """Resultado académico: el softmax no es un riesgo clínico calibrado."""

    etiqueta: str
    confianza: float
    probabilidades: dict[str, float]
    heatmap: Image.Image | None = None


def nombre_legible(etiqueta: str) -> str:
    """Traduce etiquetas del modelo externo."""
    return ETIQUETAS.get(etiqueta.strip().lower(), etiqueta.replace("_", " ").capitalize())


def interpretar_probabilidades(probabilidades, etiquetas: dict[int, str]) -> Resultado:
    """Valida forma/rango y devuelve un resultado con las tres puntuaciones."""
    valores = np.asarray(probabilidades, dtype=float)
    if valores.ndim != 1 or len(valores) != len(etiquetas):
        raise ValueError("Las probabilidades no coinciden con las etiquetas del modelo")
    if any(i not in etiquetas for i in range(len(valores))):
        raise ValueError("Índices de clases no consecutivos")
    if (
        not np.all(np.isfinite(valores))
        or np.any(valores < 0)
        or not np.isclose(valores.sum(), 1, atol=0.01)
    ):
        raise ValueError("Vector de probabilidades inválido")
    indice = int(np.argmax(valores))
    return Resultado(
        etiqueta=nombre_legible(etiquetas[indice]),
        confianza=float(valores[indice]),
        probabilidades={
            nombre_legible(etiquetas[i]): float(valor) for i, valor in enumerate(valores)
        },
    )


def preparar_tensor_alternativo(imagen: Image.Image):
    """Adaptación al contrato de ConvNeXt: RGB 224×224 y normalización ImageNet.

    Se aplica *después* del preprocesamiento académico 512×512. La eficacia
    del modelo externo con este pipeline necesita validación adicional.
    """
    import torch

    imagen = imagen.convert("RGB")
    ancho, alto = imagen.size
    if min(ancho, alto) <= 0:
        raise ValueError("Imagen sin dimensiones válidas")
    factor = 255 / min(ancho, alto)
    imagen = imagen.resize((int(ancho * factor), int(alto * factor)), Image.Resampling.BILINEAR)
    ancho, alto = imagen.size
    izquierda, arriba = (ancho - 224) // 2, (alto - 224) // 2
    imagen = imagen.crop((izquierda, arriba, izquierda + 224, arriba + 224))
    matriz = np.asarray(imagen, dtype=np.float32) / 255.0
    matriz = (matriz - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(matriz.transpose(2, 0, 1)).unsqueeze(0)


def predecir(imagen: Image.Image, generar_mapa: bool = True) -> Resultado:
    """Ejecuta inferencia con pesos reales, sin generar puntuaciones ficticias."""
    import torch

    matriz = preprocess_image(imagen)
    gris = (matriz[0, :, :, 0] * 255).astype(np.uint8)
    imagen_procesada = Image.fromarray(gris).convert("RGB")
    tensor = preparar_tensor_alternativo(imagen_procesada)
    modelo = cargar_modelo()
    with torch.inference_mode():
        logits = modelo(pixel_values=tensor).logits
        probabilidades = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    etiquetas = {int(k): v for k, v in modelo.config.id2label.items()}
    resultado = interpretar_probabilidades(probabilidades, etiquetas)
    if not generar_mapa:
        return resultado
    # Grad-CAM es un extra; si falla, se conserva la predicción.
    try:
        indice = int(np.argmax(probabilidades))
        cam = grad_cam(modelo, tensor, indice, imagen.size)
        overlay = superponer_mapa(imagen, cam)
    except (AttributeError, KeyError, RuntimeError, ValueError):
        overlay = None
    return Resultado(resultado.etiqueta, resultado.confianza, resultado.probabilidades, overlay)


def cargar_para_visualizacion(contenido: bytes, nombre_archivo: str) -> Image.Image:
    """Carga una radiografía desde bytes para mostrarla sin tocar el modelo."""
    return cargar_imagen(contenido, nombre_archivo)


def analizar_imagen(imagen: Image.Image, generar_mapa: bool = True, motor=None) -> Resultado:
    """Inversión de dependencias: admite motor falso durante pruebas unitarias."""
    predictor = motor if motor is not None else predecir
    return predictor(imagen, generar_mapa=generar_mapa)


def procesar_radiografia(
    contenido: bytes, nombre_archivo: str, generar_mapa: bool = True, motor=None
):
    """Orquesta carga y predicción sin conocer Streamlit ni rutas de disco."""
    imagen = cargar_para_visualizacion(contenido, nombre_archivo)
    resultado = analizar_imagen(imagen, generar_mapa=generar_mapa, motor=motor)
    return imagen, resultado
