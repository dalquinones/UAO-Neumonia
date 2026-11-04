"""Grad-CAM y generación de la superposición visual (sin lógica de interfaz)."""

import numpy as np
from PIL import Image


def grad_cam(modelo, tensor, indice: int, tamano: tuple[int, int]) -> np.ndarray:
    """Grad-CAM para ConvNeXtV2; si se cambia el modelo, revisar la capa objetivo."""
    import torch
    import torch.nn.functional as f

    etapas = modelo.convnextv2.encoder.stages
    activaciones = {}

    def guardar(_modulo, _entrada, salida):
        dato = salida[0] if isinstance(salida, (tuple, list)) else salida
        dato.retain_grad()
        activaciones["dato"] = dato

    hook = etapas[-1].register_forward_hook(guardar)
    try:
        with torch.enable_grad():
            modelo.zero_grad(set_to_none=True)
            logits = modelo(pixel_values=tensor).logits
            logits[0, indice].backward()
            mapa = activaciones["dato"]
            pesos = mapa.grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((pesos * mapa).sum(dim=1, keepdim=True))
            cam = f.interpolate(
                cam, size=(tamano[1], tamano[0]), mode="bilinear", align_corners=False
            )
            cam = cam[0, 0].detach().cpu().numpy()
            return (cam - cam.min()) / (np.ptp(cam) + 1e-8)
    finally:
        hook.remove()
        modelo.zero_grad(set_to_none=True)


def superponer_mapa(imagen: Image.Image, cam: np.ndarray) -> Image.Image:
    """Superpone la saliencia de la red sobre la imagen original sin modificarla."""
    base = np.asarray(imagen.convert("RGB"), dtype=np.float32)
    if cam.shape != base.shape[:2]:
        raise ValueError("El mapa de calor no corresponde al tamaño de la imagen")
    calor = np.zeros_like(base)
    calor[..., 0] = 255 * cam
    calor[..., 1] = 255 * np.clip(2 * cam - 0.5, 0, 1)
    calor[..., 2] = 255 * (1 - cam)
    mezcla = np.clip(base * 0.65 + calor * 0.35, 0, 255).astype(np.uint8)
    return Image.fromarray(mezcla)
