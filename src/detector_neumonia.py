#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
INTERFAZ GRÁFICA DE USUARIO (GUI) - DETECTOR DE NEUMONÍA
ARQUITECTURA: Presentación Desacoplada (Clean Architecture)
RESPONSABILIDAD ÚNICA:
    Interacción con el usuario, visualización de radiografías, despliegue de
    resultados clínicos y exportación de reportes (PDF y CSV).
    
    Delega toda la lógica de lectura a 'src.read_img' y la inferencia
    con explicabilidad Grad-CAM a 'src.integrator'.
=============================================================================
"""

import os
import csv
import datetime
import uuid
import numpy as np
import cv2
from tkinter import Tk, StringVar, END
from tkinter import ttk, filedialog
from tkinter.messagebox import askokcancel, showinfo, showerror, showwarning
from PIL import ImageTk, Image, ImageDraw, ImageFont

# Módulos desacoplados del sistema
try:
    from src.read_img import read_file
    from src.integrator import predict
except ModuleNotFoundError:
    from read_img import read_file
    from integrator import predict


# =============================================================================
# CONSTANTES DE SISTEMA DE DISEÑO MÉDICO
# =============================================================================
COLORS = {
    "primary": "#1E3A5F",      # Azul marino clínico - confianza y estructura
    "secondary": "#2B78E4",    # Azul de acción - botón predecir
    "danger": "#C0392B",       # Rojo sobrio - botón borrar y advertencias
    "success": "#27AE60",      # Verde médico - diagnósticos favorables
    "bg": "#F4F6F9",           # Fondo claro hospitalario
    "surface": "#FFFFFF",      # Superficie blanca de paneles y tarjetas
    "text": "#2C3E50",         # Gris oscuro neutro para texto
    "text_muted": "#7F8C8D",   # Gris secundario para subtítulos y placeholder
    "border": "#D5DBDB",       # Borde sutil
}


class App:
    def __init__(self):
        self.root = Tk()
        self.root.title("Sistema de Apoyo al Diagnóstico de Neumonía (IA)")
        self.root.geometry("860x650")
        self.root.minsize(800, 600)
        self.root.configure(bg=COLORS["bg"])

        # Configuración del sistema de estilos con ttk.Style
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self._configure_styles()

        # Variables de estado
        self.patient_id_var = StringVar()
        self.result_var = StringVar()
        self.proba_var = StringVar()
        self.label = ""
        self.proba = 0.0
        self.array = None
        self.heatmap = None
        self.report_id = 0
        self.img1_tk = None
        self.img2_tk = None

        # Contenedor principal con grid responsive
        self._build_layout()

        # Atajos de teclado (Accesibilidad)
        self._bind_shortcuts()

        self.entry_patient_id.focus_set()
        self.root.mainloop()

    def _configure_styles(self):
        """Configura la paleta de colores coherente y la escala tipográfica."""
        self.style.configure(".", background=COLORS["bg"], foreground=COLORS["text"])
        self.style.configure("TFrame", background=COLORS["bg"])
        self.style.configure("Card.TFrame", background=COLORS["surface"], relief="ridge")

        # Tipografía y etiquetas
        self.style.configure("TLabel", font=("DejaVu Sans", 10), background=COLORS["bg"], foreground=COLORS["text"])
        self.style.configure("Header.TLabel", font=("DejaVu Sans", 13, "bold"), background=COLORS["primary"], foreground="#FFFFFF")
        self.style.configure("Subtitle.TLabel", font=("DejaVu Sans", 10, "bold"), background=COLORS["surface"], foreground=COLORS["primary"])
        self.style.configure("Empty.TLabel", font=("DejaVu Sans", 9, "italic"), background=COLORS["surface"], foreground=COLORS["text_muted"])
        self.style.configure("Field.TLabel", font=("DejaVu Sans", 10, "bold"), background=COLORS["bg"], foreground=COLORS["primary"])

        # Botones
        self.style.configure("TButton", font=("DejaVu Sans", 10), padding=6)
        self.style.configure("Primary.TButton", font=("DejaVu Sans", 10, "bold"), background=COLORS["secondary"], foreground="#FFFFFF")
        self.style.map("Primary.TButton", background=[("active", "#1A5276"), ("disabled", "#BDC3C7")])

        self.style.configure("Danger.TButton", font=("DejaVu Sans", 10), background=COLORS["danger"], foreground="#FFFFFF")
        self.style.map("Danger.TButton", background=[("active", "#922B21")])

        self.style.configure("Secondary.TButton", font=("DejaVu Sans", 10), background="#E0E6ED")

    def _build_layout(self):
        """Construye la interfaz completa usando Grid Layout manager responsivo."""
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)  # La zona de imágenes se expande

        # --- 1. BANNER INSTITUCIONAL SUPERIOR ---
        header_frame = ttk.Frame(self.root, style="Card.TFrame", padding=12)
        header_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=(12, 8))
        header_frame.columnconfigure(0, weight=1)

        self.lab_title = ttk.Label(
            header_frame,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            style="Header.TLabel",
            anchor="center",
            padding=8,
        )
        self.lab_title.grid(row=0, column=0, sticky="ew")

        # --- 2. ÁREA CENTRAL: PANELES DE IMÁGENES ---
        images_frame = ttk.Frame(self.root, style="TFrame")
        images_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=6)
        images_frame.columnconfigure(0, weight=1, uniform="img_col")
        images_frame.columnconfigure(1, weight=1, uniform="img_col")
        images_frame.rowconfigure(1, weight=1)

        # Panel Radiografía Original
        card_img1 = ttk.LabelFrame(images_frame, text=" Radiografía Original ", padding=8)
        card_img1.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=4)
        card_img1.columnconfigure(0, weight=1)
        card_img1.rowconfigure(0, weight=1)

        self.lab1 = ttk.Label(card_img1, text="Imagen Radiográfica", style="Subtitle.TLabel")
        self.label_img1 = ttk.Label(
            card_img1,
            text="[ Sin radiografía cargada ]\n\nPresione 'Cargar Imagen' o use Ctrl+O",
            style="Empty.TLabel",
            anchor="center",
            justify="center",
        )
        self.label_img1.grid(row=0, column=0, sticky="nsew")

        # Panel Grad-CAM
        card_img2 = ttk.LabelFrame(images_frame, text=" Mapa de Activación (Grad-CAM) ", padding=8)
        card_img2.grid(row=0, column=1, sticky="nsew", padx=(8, 0), pady=4)
        card_img2.columnconfigure(0, weight=1)
        card_img2.rowconfigure(0, weight=1)

        self.lab2 = ttk.Label(card_img2, text="Mapa de Activación (Grad-CAM)", style="Subtitle.TLabel")
        self.label_img2 = ttk.Label(
            card_img2,
            text="[ Sin mapa de activación ]\n\nEl análisis Grad-CAM aparecerá tras la predicción",
            style="Empty.TLabel",
            anchor="center",
            justify="center",
        )
        self.label_img2.grid(row=0, column=0, sticky="nsew")

        # Compatibilidad con tests anteriores que buscan self.text_img1 y self.text_img2
        self.text_img1 = self.label_img1
        self.text_img2 = self.label_img2

        # --- 3. BARRA DE PROGRESO DE INFERENCIA ---
        self.progress = ttk.Progressbar(self.root, mode="indeterminate", length=400)
        self.progress.grid(row=2, column=0, sticky="ew", padx=25, pady=2)

        # --- 4. ZONA DE CAMPOS CLÍNICOS Y RESULTADOS ---
        fields_frame = ttk.Frame(self.root, style="TFrame", padding=6)
        fields_frame.grid(row=3, column=0, sticky="ew", padx=15, pady=4)
        fields_frame.columnconfigure(1, weight=1)
        fields_frame.columnconfigure(3, weight=1)
        fields_frame.columnconfigure(5, weight=1)

        # Cédula Paciente
        self.lab4 = ttk.Label(fields_frame, text="Cédula Paciente:", style="Field.TLabel")
        self.lab4.grid(row=0, column=0, padx=(0, 6), sticky="w")
        self.entry_patient_id = ttk.Entry(fields_frame, textvariable=self.patient_id_var, width=16, font=("DejaVu Sans", 10))
        self.entry_patient_id.grid(row=0, column=1, padx=(0, 15), sticky="w")

        # Diagnóstico Resultante
        self.lab3 = ttk.Label(fields_frame, text="Resultado:", style="Field.TLabel")
        self.lab3.grid(row=0, column=2, padx=(10, 6), sticky="w")
        self.entry_result = ttk.Entry(
            fields_frame,
            textvariable=self.result_var,
            width=18,
            font=("DejaVu Sans", 10, "bold"),
            state="readonly",
            justify="center",
        )
        self.entry_result.grid(row=0, column=3, padx=(0, 15), sticky="w")
        self.text_result = self.entry_result

        # Certeza / Probabilidad
        self.lab6 = ttk.Label(fields_frame, text="Probabilidad:", style="Field.TLabel")
        self.lab6.grid(row=0, column=4, padx=(10, 6), sticky="w")
        self.entry_proba = ttk.Entry(
            fields_frame,
            textvariable=self.proba_var,
            width=12,
            font=("DejaVu Sans", 10, "bold"),
            state="readonly",
            justify="center",
        )
        self.entry_proba.grid(row=0, column=5, sticky="w")
        self.text_proba = self.entry_proba

        # --- 5. BARRA INFERIOR DE ACCIONES (BOTONES) ---
        actions_frame = ttk.Frame(self.root, style="TFrame", padding=10)
        actions_frame.grid(row=4, column=0, sticky="ew", padx=15, pady=(4, 12))
        for col in range(5):
            actions_frame.columnconfigure(col, weight=1)

        self.btn_load = ttk.Button(actions_frame, text="📁 Cargar Imagen", style="Secondary.TButton", command=self.load_image)
        self.btn_load.grid(row=0, column=0, padx=4, sticky="ew")

        self.btn_predict = ttk.Button(actions_frame, text="⚡ Predecir", style="Primary.TButton", state="disabled", command=self.run_prediction)
        self.btn_predict.grid(row=0, column=1, padx=4, sticky="ew")

        self.btn_save = ttk.Button(actions_frame, text="💾 Guardar CSV", style="Secondary.TButton", state="disabled", command=self.save_csv)
        self.btn_save.grid(row=0, column=2, padx=4, sticky="ew")

        self.btn_pdf = ttk.Button(actions_frame, text="📄 Generar PDF", style="Secondary.TButton", state="disabled", command=self.create_pdf)
        self.btn_pdf.grid(row=0, column=3, padx=4, sticky="ew")

        self.btn_delete = ttk.Button(actions_frame, text="🗑️ Borrar", style="Danger.TButton", command=self.reset_form)
        self.btn_delete.grid(row=0, column=4, padx=4, sticky="ew")

    def _bind_shortcuts(self):
        """Enlaza atajos de teclado para navegación y accesibilidad médica."""
        self.root.bind("<Control-o>", lambda e: self.load_image())
        self.root.bind("<Control-O>", lambda e: self.load_image())
        self.root.bind("<Return>", lambda e: self._on_enter_pressed())
        self.root.bind("<Control-s>", lambda e: self.save_csv())
        self.root.bind("<Control-S>", lambda e: self.save_csv())
        self.root.bind("<Escape>", lambda e: self.reset_form())

    def _on_enter_pressed(self):
        """Dispara la predicción al presionar Enter si hay imagen cargada."""
        if str(self.btn_predict["state"]) == "normal":
            self.run_prediction()

    # ==================== MÉTODOS DE LA APLICACIÓN ====================

    def load_image(self):
        """Carga una imagen radiográfica (.dcm, .jpg, .png) y la proyecta con manejo de excepciones."""
        filepath = filedialog.askopenfilename(
            title="Seleccionar Radiografía",
            filetypes=(
                ("Archivos Médicos", "*.dcm *.jpeg *.jpg *.png"),
                ("DICOM (*.dcm)", "*.dcm"),
                ("JPEG/JPG (*.jpg;*.jpeg)", "*.jpg *.jpeg"),
                ("PNG (*.png)", "*.png"),
            ),
        )
        if not filepath:
            return  # Usuario canceló el diálogo

        try:
            # Delegar la lectura al Módulo 1 (read_img)
            self.array, img2show = read_file(filepath)
            
            # Redimensionar para mostrar en pantalla
            pil_thumb = img2show.resize((260, 260), Image.Resampling.LANCZOS)
            self.img1_tk = ImageTk.PhotoImage(pil_thumb)
            # Limpiar estado anterior para evitar consistencia cruzada (Devin Review)
            self.result_var.set("")
            self.proba_var.set("")
            self.label_img1.configure(image=self.img1_tk, text="")
            self.label_img2.configure(image="", text="[ Sin mapa de activación ]\n\nEl análisis Grad-CAM aparecerá tras la predicción")
            self.img2_tk = None
            self.heatmap = None
            self.label = ""
            self.proba = 0.0
            
            self.btn_predict["state"] = "normal"
            self.btn_save["state"] = "disabled"
            self.btn_pdf["state"] = "disabled"
        except Exception as e:
            showerror("Error al cargar imagen", f"No se pudo procesar la radiografía:\n{str(e)}")

    def run_prediction(self):
        """Ejecuta la inferencia delegando la orquestación a 'integrator.py' con feedback visual."""
        if self.array is None:
            showwarning("Sin imagen", "Primero cargue una radiografía médica para predecir.")
            return

        try:
            # Animar barra de progreso durante la inferencia
            self.progress.start(10)
            self.root.update_idletasks()

            # Llamada de bajo acoplamiento al integrador
            self.label, self.proba, self.heatmap = predict(self.array)

            # Mostrar mapa de calor en la interfaz
            heat_pil = Image.fromarray(self.heatmap).resize((260, 260), Image.Resampling.LANCZOS)
            self.img2_tk = ImageTk.PhotoImage(heat_pil)
            self.label_img2.configure(image=self.img2_tk, text="")

            # Mostrar valores diagnósticos en variables reactivas
            self.result_var.set(self.label.upper())
            self.proba_var.set(f"{self.proba:.2f}%")
            self.btn_save["state"] = "normal"
            self.btn_pdf["state"] = "normal"

        except Exception as e:
            showerror("Error de Inferencia", f"Ocurrió una falla durante el análisis Grad-CAM:\n{str(e)}")
        finally:
            self.progress.stop()

    def save_csv(self):
        """Almacena el historial diagnóstico en un archivo CSV."""
        if not self.label:
            showwarning("Atención", "No hay diagnóstico disponible para guardar. Realice una predicción primero.")
            return
        with open("historial.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            patient_id = self.entry_patient_id.get().strip() or "S_N"
            writer.writerow([patient_id, fecha, self.label, f"{self.proba:.2f}%"])
        showinfo(title="Guardar", message="Los datos del paciente se guardaron con éxito en historial.csv.")

    def create_pdf(self):
        """Genera un reporte médico clínico en formato PDF de forma nativa e independiente del SO."""
        if not self.label:
            showwarning("Atención", "No hay un diagnóstico válido para exportar. Realice una predicción primero.")
            return

        w, h = 1024, 768
        report = Image.new("RGB", (w, h), color=(255, 255, 255))
        draw = ImageDraw.Draw(report)

        # Cargar fuentes del sistema o usar la predeterminada si no están disponibles
        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            font_info = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
            font_labels = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except Exception:
            font_title = font_info = font_labels = ImageFont.load_default()

        # Encabezado institucional
        draw.rectangle([(0, 0), (w, 80)], fill=(30, 42, 56))
        draw.text((30, 26), "REPORTE MEDICO - DETECCION DE NEUMONIA ASISTIDA POR IA", fill=(255, 255, 255), font=font_title)

        # Información del paciente
        fecha_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        patient_id = self.entry_patient_id.get().strip() or "S_N"
        draw.text((30, 100), f"ID Paciente: {patient_id}    |    Fecha: {fecha_str}", fill=(20, 20, 20), font=font_info)
        draw.text((30, 130), f"Diagnostico: {self.label.upper()}    |    Certeza Medica: {self.proba:.2f}%", fill=(20, 20, 20), font=font_info)

        # Pegar radiografía original
        if self.array is not None:
            img_orig = self.array.copy()
            if len(img_orig.shape) == 2:
                img_orig = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2RGB)
            elif img_orig.shape[2] == 1:
                img_orig = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2RGB)
            pil_orig = Image.fromarray(img_orig).resize((420, 420))
            report.paste(pil_orig, (50, 180))
            draw.text((50, 615), "Radiografia Original", fill=(50, 50, 50), font=font_labels)

        # Pegar mapa de calor
        if self.heatmap is not None:
            pil_heat = Image.fromarray(self.heatmap).resize((420, 420))
            report.paste(pil_heat, (540, 180))
            draw.text((540, 615), "Explicabilidad Grad-CAM (Region Pulmonar)", fill=(50, 50, 50), font=font_labels)

        # Generar nombre único usando timestamp
        patient_id = self.patient_id_var.get().strip() or "S_N"
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_suffix = str(uuid.uuid4())[:6]
        pdf_path = f"Reporte_{patient_id}_{timestamp_str}_{unique_suffix}.pdf"
        report.save(pdf_path)
        showinfo(title="PDF Generado", message=f"Reporte clínico generado con éxito:\n{pdf_path}")

    def reset_form(self):
        """Limpia todos los campos para un nuevo análisis."""
        if askokcancel(title="Confirmar", message="¿Desea limpiar todos los campos del formulario?"):
            self.entry_patient_id.delete(0, END)
            self.result_var.set("")
            self.proba_var.set("")
            self.label_img1.configure(image="", text="[ Sin radiografía cargada ]\n\nPresione 'Cargar Imagen' o use Ctrl+O")
            self.label_img2.configure(image="", text="[ Sin mapa de activación ]\n\nEl análisis Grad-CAM aparecerá tras la predicción")
            self.img1_tk = None
            self.img2_tk = None
            self.array = None
            self.heatmap = None
            self.label = ""
            self.proba = 0.0
            self.btn_predict["state"] = "disabled"
            self.btn_save["state"] = "disabled"
            self.btn_pdf["state"] = "disabled"


def main():
    App()
    return 0


if __name__ == "__main__":
    main()

