# Plan de Mejora — Interfaz Gráfica de Neumonía

## Contexto

La aplicación detecta neumonía en radiografías de tórax usando CNN + Grad-CAM. La interfaz actual usa Tkinter con posicionamiento absoluto (`place()`) y widgets genéricos, lo que genera una experiencia frágil y sin identidad visual.

## Problemas identificados

### 1. Layout rígido
- Posicionamiento absoluto en `detector_neumonia.py:79-97` (`place(x, y)`)
- Ventana fija 815×560, no responsive
- Cambiar tamaño rompe la UI

### 2. Widgets inadecuados
- `Text` widgets para mostrar imágenes (`detector_neumonia.py:66-67`) — usar `Label` con `ImageTk`
- `Text` para resultados diagnósticos — usar `Label` o `Entry` con estado readonly

### 3. Sin jerarquía visual
- Todos los labels usan `font_bold` sin variación
- Título compite con subtítulos de imágenes
- Botones primarios (Predecir) y secundarios (Borrar) son idénticos

### 4. Sin estados de feedback
- "Predecir" deshabilitado sin indicación visual clara
- Sin progress bar durante inferencia (~2-5 segundos)
- Errores solo en `try/except` interno, sin UI

### 5. Sin sistema de diseño
- No hay `ttk.Style()` configurado
- Colores del sistema operativo, sin paleta definida
- Fuentes del sistema sin escala tipográfica

---

## Plan de mejora

### Fase 1: Reemplazar posicionamiento absoluto con layout manager

**Objetivo:** Layout que se adapte al contenido y soporte redimensionamiento.

**Cambios en `detector_neumonia.py`:**
```python
# Reemplazar place() con grid()
self.root.columnconfigure(0, weight=1)
self.root.columnconfigure(1, weight=1)

self.lab_title.grid(row=0, column=0, columnspan=2, pady=10)
self.lab1.grid(row=1, column=0)
self.lab2.grid(row=1, column=1)
self.text_img1.grid(row=2, column=0, padx=10)
self.text_img2.grid(row=2, column=1, padx=10)
# ... resto de widgets
```

**Resultado:** Ventana resizeable, widgets reposicionan automáticamente.

---

### Fase 2: Corregir widgets de imagen

**Objetivo:** Uso correcto de widgets de Tkinter.

**Cambios:**
```python
# Antes (incorrecto)
self.text_img1 = Text(self.root, width=31, height=15)

# Después (correcto)
self.label_img1 = ttk.Label(self.root, anchor="center")
self.label_img2 = ttk.Label(self.root, anchor="center")
```

**Actualizar métodos:**
- `load_image()`: usar `self.label_img1.configure(image=self.img1_tk)`
- `run_prediction()`: usar `self.label_img2.configure(image=self.img2_tk)`
- Eliminar `text_img1.delete()` y `text_img2.delete()`

---

### Fase 3: Sistema de estilos con ttk.Style

**Objetivo:** Paleta de colores coherente y escala tipográfica.

**Implementar al inicio de `App.__init__`:**
```python
style = ttk.Style()
style.theme_use("clam")  # Base consistente cross-platform

# Colores médicos — azul confianza, gris neutro, rojo alerta
COLORS = {
    "primary": "#1E3A5F",    # Azul oscuro — confianza médica
    "secondary": "#4A90D9",  # Azul claro — acciones
    "danger": "#C0392B",     # Rojo — alertas
    "bg": "#F5F6FA",         # Gris claro — fondo
    "text": "#2C3E50",       # Casi negro — texto
}

style.configure("TLabel", font=("Segoe UI", 11), foreground=COLORS["text"])
style.configure("Title.TLabel", font=("Segoe UI", 14, "bold"), foreground=COLORS["primary"])
style.configure("TButton", font=("Segoe UI", 10), padding=6)
style.configure("Primary.TButton", background=COLORS["secondary"], foreground="white")
style.configure("Danger.TButton", background=COLORS["danger"], foreground="white")
```

---

### Fase 4: Jerarquía visual y estados

**Objetivo:** Distinguir acciones primarias/secundarias y dar feedback.

**Cambios:**
```python
# Botón primario (Predecir)
self.btn_predict = ttk.Button(
    self.root, text="Predecir", style="Primary.TButton",
    state="disabled", command=self.run_prediction
)

# Botón secundario (Borrar)
self.btn_delete = ttk.Button(
    self.root, text="Borrar", style="Danger.TButton",
    command=self.reset_form
)

# Progress bar durante inferencia
self.progress = ttk.Progressbar(self.root, mode="indeterminate")

# En run_prediction():
self.progress.start(10)  # Animar
self.label, self.proba, self.heatmap = predict(self.array)
self.progress.stop()
```

---

### Fase 5: Feedback de errores y empty states

**Objetivo:** Mostrar errores y estados vacíos al usuario.

**Implementar:**
```python
def load_image(self):
    filepath = filedialog.askopenfilename(...)
    if not filepath:
        return  # Usuario canceló — no mostrar error

    try:
        self.array, img2show = read_file(filepath)
    except Exception as e:
        showerror("Error", f"No se pudo cargar la imagen:\n{str(e)}")
        return

    # ... resto de lógica

def run_prediction(self):
    if self.array is None:
        showwarning("Sin imagen", "Primero cargue una radiografía.")
        return

    # ... inferencia
```

**Empty state en labels de imagen:**
```python
self.label_img1.configure(text="Arrastre o seleccione una radiografía")
self.label_img2.configure(text="Resultado del Grad-CAM aparecerá aquí")
```

---

### Fase 6: Mejoras de accesibilidad

**Objetivo:** Navegación por teclado y contraste adecuado.

**Cambios:**
```python
# Shortcut keys
self.root.bind("<Control-o>", lambda e: self.load_image())
self.root.bind("<Return>", lambda e: self.run_prediction())
self.root.bind("<Control-s>", lambda e: self.save_csv())
self.root.bind("<Escape>", lambda e: self.reset_form())

# Tooltips (con customtkinter o implementación simple)
# Focus visible en botones
style.configure("TButton", focuscolor=COLORS["secondary"])
```

---

## Archivos a modificar

| Archivo | Cambios |
|---------|---------|
| `src/detector_neumonia.py` | Reemplazar `place()` con `grid()`, corregir widgets, añadir `ttk.Style`, progress bar, manejo de errores, shortcuts |

## Archivos NO modificar

| Archivo | Razón |
|---------|-------|
| `src/read_img.py` | Lógica de lectura — sin cambios |
| `src/integrator.py` | Orquestación — sin cambios |
| `src/grad_cam.py` | Inferencia — sin cambios |
| `src/preprocess_img.py` | Preprocesamiento — sin cambios |

---

## Orden de ejecución

1. **Fase 1** → Layout con `grid()` (base para todo lo demás)
2. **Fase 2** → Corregir widgets de imagen
3. **Fase 3** → Sistema de estilos
4. **Fase 4** → Jerarquía visual y progress bar
5. **Fase 5** → Manejo de errores y empty states
6. **Fase 6** → Accesibilidad y shortcuts

## Criterios de éxito

- [x] Ventana resizeable sin romper layout (Grid responsive con minsize 800x600)
- [x] Imágenes en `Label` en vez de `Text` (ttk.Label con empty state descriptivo)
- [x] Paleta de colores coherente (ttk.Style con colores clínicos institucionales)
- [x] Botón "Predecir" visualmente distintivo (Primary.TButton con estados dinámicos)
- [x] Progress bar durante inferencia (ttk.Progressbar indeterminada)
- [x] Mensajes de error claros al usuario (try/except con showerror/showwarning)
- [x] Shortcuts de teclado funcionales (Ctrl+O, Enter, Ctrl+S, Esc)