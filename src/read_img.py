"""
=============================================================================
MÓDULO 1: read_img.py
ARQUITECTURA: Alta Cohesión y Bajo Acoplamiento 
RESPONSABILIDAD ÚNICA:
    Lectura, decodificación y validación de integridad de archivos de imagen 
    médica en formatos DICOM (.dcm) y estándar (.jpg, .jpeg, .png).

OBJETIVOS ESPECÍFICOS POR FUNCIÓN:

1. read_dicom_file(path: str) -> tuple[np.ndarray, Image.Image]:
   - Objetivo: Procesar archivos médicos de estándar clínico DICOM.
   - Operaciones:
       a) Valida la existencia física del archivo.
       b) Extrae el 'pixel_array' de 12/16 bits con pydicom.
       c) Aplica reescalamiento a 8 bits (0-255) eliminando ruidos negativos.
       d) Convierte de 1 canal (monocromático) a 3 canales (RGB).
   - Retorna: (matriz_numpy_rgb, objeto_pil_para_interfaz).
   - Excepciones: FileNotFoundError si no existe la ruta.

2. read_jpg_file(path: str) -> tuple[np.ndarray, Image.Image]:
   - Objetivo: Procesar imágenes comprimidas comunes (JPG, PNG).
   - Operaciones:
       a) Valida la existencia del archivo en disco.
       b) Decodifica la imagen con OpenCV (formato BGR).
       c) Realiza la corrección de espacio de color BGR -> RGB.
       d) Genera el objeto visual PIL.
   - Retorna: (matriz_numpy_rgb, objeto_pil_para_interfaz).
   - Excepciones: FileNotFoundError (no existe) y ValueError (archivo corrupto).

3. read_file(path: str) -> tuple[np.ndarray, Image.Image]:
   - Objetivo: Función fachada / enrutadora (Facade Pattern).
   - Operaciones:
       a) Inspecciona la extensión del archivo ingresado.
       b) Delega la lectura al especialista correcto (.dcm -> read_dicom_file, 
          .jpg/.png -> read_jpg_file).
       c) Ofrece tolerancia a fallos para archivos médicos que no tienen 
          extensión explícita en su nombre.
   - Retorna: El resultado homogéneo (tupla) de cualquiera de los dos lectores.

BENEFICIOS DE DISEÑO:
- Desacoplamiento total de la UI: No importa Tkinter, Qt ni Streamlit.
- Desacoplamiento del Modelo: No sabe qué es TensorFlow ni cómo se procesa.
- Facilidad para Pruebas (Testability): Permite escribir pruebas unitarias 
  con pytest pasando rutas válidas, rutas falsas y archivos corruptos.
=============================================================================
"""


import os
import cv2
import numpy as np
import pydicom as dicom
from PIL import Image


def read_dicom_file(path: str) -> tuple[np.ndarray, Image.Image]:
    """
    Lee un archivo en formato estándar de radiología médica (.dcm).
    
    Args:
        path (str): Ruta absoluta o relativa al archivo DICOM.
        
    Returns:
        tuple[np.ndarray, Image.Image]: Una tupla que contiene:
            - np.ndarray: Matriz de imagen RGB en formato NumPy (para cálculos).
            - Image.Image: Objeto de imagen PIL (para renderizado en UI).
            
    Raises:
        FileNotFoundError: Si el archivo especificado no existe en la ruta.
    """
    # 1. Validación de seguridad: Verificar que el archivo realmente existe en el disco
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: No se encontró el archivo DICOM en la ruta: {path}")

    # 2. Leer los metadatos y la información binaria del archivo DICOM
    dcm = dicom.dcmread(path)

    # 3. Extraer la matriz numérica de píxeles (pixel_array) y convertir a punto flotante
    #    para evitar desbordamientos durante operaciones matemáticas.
    img_array = dcm.pixel_array.astype(float)

    # 4. Normalización de intensidad (Escalamiento a 8 bits: 0 a 255):
    #    Las radiografías DICOM suelen venir en 12 o 16 bits (valores de 0 a 4095).
    #    np.maximum(..., 0) elimina posibles valores negativos (ruido del sensor).
    max_val = img_array.max()
    if max_val > 0:
        img_normalized = (np.maximum(img_array, 0) / max_val) * 255.0
    else:
        img_normalized = np.zeros_like(img_array)

    # 5. Convertir a entero sin signo de 8 bits (tipo estándar que usan OpenCV y Pillow)
    img_uint8 = np.uint8(img_normalized)

    # 6. Garantizar que la imagen tenga 3 canales (R, G, B):
    #    Las radiografías son monocromáticas (2 dimensiones: alto x ancho).
    #    Las convertimos a 3 canales para que sean compatibles con la interfaz y OpenCV.
    if len(img_uint8.shape) == 2:
        img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_uint8

    # 7. Crear el objeto PIL para que la interfaz gráfica (Tkinter) pueda redimensionarla y mostrarla
    img2show = Image.fromarray(img_rgb)

    # Retorna tanto la matriz NumPy (para cálculos) como el objeto PIL (para la pantalla)
    return img_rgb, img2show


def read_jpg_file(path: str) -> tuple[np.ndarray, Image.Image]:
    """
    Lee una imagen fotográfica estándar (.jpg, .jpeg, .png).
    
    Args:
        path (str): Ruta absoluta o relativa al archivo de imagen.
        
    Returns:
        tuple[np.ndarray, Image.Image]: Una tupla que contiene:
            - np.ndarray: Matriz de imagen RGB en formato NumPy (para cálculos).
            - Image.Image: Objeto de imagen PIL (para renderizado en UI).
            
    Raises:
        FileNotFoundError: Si el archivo especificado no existe en la ruta.
        ValueError: Si el archivo está dañado o no puede ser decodificado por OpenCV.
    """
    # 1. Validación de existencia
    if not os.path.exists(path):
        raise FileNotFoundError(f"Error: No se encontró la imagen en la ruta: {path}")

    # 2. OpenCV lee imágenes por defecto en formato BGR (Azul, Verde, Rojo)
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise ValueError(f"Error: El archivo está dañado o no es una imagen válida: {path}")

    # 3. Convertir de BGR a RGB para que los colores se vean naturales en pantalla
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 4. Crear objeto PIL para la interfaz
    img2show = Image.fromarray(img_rgb)

    return img_rgb, img2show


def read_file(path: str) -> tuple[np.ndarray, Image.Image]:
    """
    Función orquestadora inteligente para lectura de imágenes médicas.
    
    Detecta automáticamente la extensión del archivo y delega la lectura
    a la función especializada correspondiente. Proporciona tolerancia a fallos
    para archivos médicos que carecen de extensión explícita.
    
    Args:
        path (str): Ruta absoluta o relativa al archivo a leer.
        
    Returns:
        tuple[np.ndarray, Image.Image]: Una tupla con la matriz NumPy y el objeto PIL.
        
    Raises:
        FileNotFoundError: Si el archivo especificado no existe en la ruta.
        ValueError: Si el archivo es inválido o no soportado.
    """
    # Obtiene la extensión del archivo en minúsculas (por ejemplo: '.dcm' o '.jpg')
    extension = os.path.splitext(path)[1].lower()

    if extension == ".dcm":
        return read_dicom_file(path)
    elif extension in [".jpg", ".jpeg", ".png"]:
        return read_jpg_file(path)
    else:
        # Si el archivo médico no tiene extensión explícita, intenta leerlo como DICOM primero
        try:
            return read_dicom_file(path)
        except Exception:
            return read_jpg_file(path)
