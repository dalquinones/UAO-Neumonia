# HANDOFF H0 → H1
Fecha: 2026-08-25 · Rama: spike/compatibilidad · Commit: N/D (sin commit registrado en este hilo) · Autor: Persona A (Tech Lead/DevOps)

## 1. ESTADO
Completado:
✅ spike_compat_legacy.py (Plan B) creado, corregido y EJECUTADO con éxito verificado
✅ Veredicto empírico obtenido: PLAN B OK
✅ spike_compat.py (Plan A) creado y corregido (código listo, NO ejecutado en este hilo)

Pendiente (TODO heredado):
⏳ Ejecutar spike_compat.py (Plan A) para confirmar/descartar si Keras 3 puro también carga el modelo
⏳ Verificar mapeo de clases 0/1/2 con imágenes reales (en este hilo solo se probó con ruido aleatorio)
⏳ Calcular SHA256 de models/conv_MLP_84.h5
⏳ Redactar formalmente docs/adr/ADR-001-keras3-vs-tfkeras.md

## 2. ARCHIVOS QUE EXISTEN AHORA
| Ruta | Estado | Firmas públicas |
|---|---|---|
| spike_compat.py | ✅ completo, sin ejecutar en este hilo | main() -> None; paso_1_versiones() -> dict[str, Any]; paso_2_y_3_cargar_modelo(ruta: Path) -> tuple[Any, str]; paso_4_summary_y_tabla(modelo: Any) -> list[dict[str, Any]]; paso_5_capas_conv2d(modelo: Any) -> str \| None; paso_6_shapes(modelo: Any) -> tuple[Any, Any]; paso_7_inferencia(modelo: Any, input_shape: Any) -> np.ndarray \| None; obtener_logits_pre_softmax(modelo: Any) -> Any; paso_8_gradcam_submodelo(modelo: Any, nombre_capa: str \| None, input_shape: Any) -> tuple[bool, bool, float]; veredicto_final(modelo_cargado: bool, submodelo_ok: bool, gradiente_no_none: bool, norma: float) -> str |
| spike_compat_legacy.py | ✅ completo y EJECUTADO (VEREDICTO: PLAN B OK) | mismas firmas que spike_compat.py |

## 3. DECISIONES TOMADAS EN ESTE HILO
- D0.1: Backend = Plan B (`TF_USE_LEGACY_KERAS=1` + paquete `tf-keras`) — Razón: cargó el modelo y produjo gradientes no nulos verificados; Plan A no se ejecutó en este hilo — Impacto: `load_model.py` y `grad_cam.py` deben fijar la variable de entorno ANTES de `import tensorflow`; `pyproject.toml` debe incluir `tf-keras`.
- D0.2: Capa objetivo Grad-CAM = `conv10_thisone` — Razón: `model.layers` no respeta orden topológico en ramas residuales; el "último Conv2D por índice" era `conv5skip`, una proyección 1x1 de shortcut sin semántica espacial comparable — Impacto: `grad_cam.py` debe fijar este nombre (o excluir capas `*skip*`).
- D0.3: Grad-CAM debe derivar sobre el logit pre-softmax, no sobre la probabilidad — Razón: con entradas casi saturadas (~one-hot) la derivada del softmax se anula numéricamente, verificado empíricamente — Impacto: `grad_cam.py` debe reconstruir el logit (kernel/bias de la última Dense) o exponer el modelo sin activación final.

## 4. VERSIONES EXACTAS CONFIRMADAS
python=3.13.14 · tensorflow=2.20.0 · keras=2.20.1 (legacy, vía tf-keras) · numpy=2.5.2 · opencv=N/D (no probado) · pydicom=N/D (no probado)

## 5. HECHOS VERIFICADOS EMPÍRICAMENTE
- GPU no detectada → evidencia: `gpu_detectada: False`, `gpus: []`
- `conv_MLP_84.h5` carga con `load_model(path)` simple, sin necesitar `compile=False` ni `safe_mode=False` → evidencia: `[OK] Exito con: load_model(path)`
- Modelo funcional "Net5Blocks", 57 capas, 9,796,483 parámetros totales (9,794,755 entrenables) → evidencia: `model.summary()` capturado
- `input_shape=(None, 512, 512, 1)`, `output_shape=(None, 3)` → evidencia: log paso 6
- 15 capas Conv2D/SeparableConv2D; `conv10_thisone` coincide por nombre exacto con convención del autor → evidencia: listado paso 5
- Con ruido uniforme aleatorio, la predicción se satura (~one-hot, clase índice 2, prob≈1.0) → evidencia: `probabilidades: [3.16e-17, 4.86e-17, 1.0]`
- Gradiente sobre probabilidad post-softmax fue numéricamente nulo (norma≈2.66e-18, bajo umbral 1e-12) → evidencia: log de la corrida previa a la corrección
- Gradiente sobre logit pre-softmax NO es nulo: norma=1.866533e-02, shape=(1,31,31,128) → evidencia: log de la corrida corregida
- Resultado final reproducible: `VEREDICTO: PLAN B OK`, `Metodo de carga exitoso: load_model(path)`

## 6. TRAMPAS / GOTCHAS DESCUBIERTOS
- `keras.__version__` lanza `AttributeError` bajo `TF_USE_LEGACY_KERAS=1` (resuelve a `tf_keras.api._v2.keras`, sin `__version__` público) → Solución: fallback a `import tf_keras; tf_keras.__version__`.
- Heurística "última Conv2D por índice de `model.layers`" eligió `conv5skip` (shortcut 1x1) en vez de `conv10_thisone` (tronco principal), porque `model.layers` no sigue orden topológico estricto en ramas residuales → Solución: excluir nombres con `"skip"` y priorizar coincidencia exacta con `"conv10_thisone"`.
- Norma de gradiente mostrada como `0.000000` por redondeo a 6 decimales, ocultando que el valor real era `~2.66e-18` (no exactamente cero) → Solución: usar notación científica (`:.6e`) y comparar contra un épsilon (`1e-12`), no contra `== 0.0`.
- Gradiente genuinamente casi nulo al derivar sobre la probabilidad softmax con entrada de ruido saturando la predicción → Solución: derivar sobre el logit pre-softmax reconstruido con `kernel`/`bias` de la última capa `Dense`.
- Caracteres UTF-8 (tildes, guion largo) corruptos en PowerShell (`v├¡a`, mojibake) → Solución: `sys.stdout.reconfigure(encoding="utf-8")` en el script + evitar tildes en cadenas impresas (solo en docstrings).
- `grep` no existe en PowerShell nativo → Solución: `Select-String -Path <archivo> -Pattern "texto"`.
- Warning de TF "Lambda layer... not present in its tracked objects" al reconstruir el logit con `tf.matmul` directo sobre tensor simbólico → Pendiente para H5: envolver `kernel`/`bias` en una capa `Dense` clonada (no operaciones TF sueltas) para cumplir la política de cero warnings.

## 7. LO QUE NECESITA EL SIGUIENTE HILO
- Debe implementar: `pyproject.toml` con `tensorflow==2.20.*`, `tf-keras`, `numpy`, `opencv-python-headless`, `pydicom`, `pillow`, `reportlab`, `pytest`, `ruff`; estructura de carpetas completa; `docs/CONTRATOS.md` congelado.
- Debe respetar estos contratos: `TF_USE_LEGACY_KERAS=1` fijado ANTES de `import tensorflow` en todo módulo que use TF (`src/load_model.py`, `src/grad_cam.py`); capa Grad-CAM = `"conv10_thisone"`; `input_shape=(None,512,512,1)`; `output_shape=(None,3)`.
- NO debe tocar: `spike_compat.py` ni `spike_compat_legacy.py` (desechables, no se copian a `src/`).
