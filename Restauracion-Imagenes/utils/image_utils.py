# utils/image_utils.py

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
from io import BytesIO
import replicate
import requests
from dotenv import load_dotenv
import streamlit as st # Importante para mostrar errores en la app

# Cargar variables de entorno (.env) para leer el token de Replicate
load_dotenv()

# ==========================================
# 1. FUNCIONES DE PDI (Procesamiento Clásico)
# ==========================================

def adjust_brightness(image: Image.Image, factor: float) -> Image.Image:
    """
    Ajusta el brillo de la imagen.
    Factor: 1.0 = Original, <1.0 = Más oscuro, >1.0 = Más brillante.
    """
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_sharpness(image: Image.Image, factor: float) -> Image.Image:
    """
    Ajusta la nitidez (Sharpening).
    Factor: 1.0 = Original, >1.0 = Bordes más marcados.
    """
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)

def equalize_histogram(image: Image.Image) -> Image.Image:
    """
    Aplica ecualización de histograma adaptativa para mejorar el contraste.
    Convierte a YUV, ecualiza el canal Y (Luminancia) y reconvierte.
    """
    # Convertir PIL a NumPy
    img = np.array(image.convert('RGB'))
    
    # Convertir a YUV
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    
    # Ecualizar el canal Y (Luminosidad)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    
    # Convertir de nuevo a RGB
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    return Image.fromarray(img_output)

def reduce_artifacts(image: Image.Image) -> Image.Image:
    """
    Reduce ruido y artefactos usando un filtro bilateral (mantiene bordes).
    """
    img = np.array(image)
    
    # Si la imagen tiene transparencia (RGBA), convertir a RGB para OpenCV
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
    # Filtro Bilateral: Suaviza texturas pero respeta bordes fuertes
    smooth = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    
    return Image.fromarray(smooth)

# ==========================================
# 2. FUNCIONES PLACEHOLDER (Para evitar errores en app.py)
# ==========================================

def extract_control_maps(image):
    """
    Función auxiliar requerida por la estructura original del proyecto.
    """
    return {'canny': None, 'depth': None}

def extract_face_embedding(image):
    """
    Función auxiliar requerida por la estructura original del proyecto.
    """
    return None

# ==========================================
# 3. IA GENERATIVA: SUPER-RESOLUCIÓN (Real-ESRGAN)
# ==========================================

def upscale_image(image: Image.Image) -> Image.Image:
    """
    Aplica Super-Resolución (4x) usando Real-ESRGAN vía Replicate API.
    Requiere REPLICATE_API_TOKEN en el archivo .env
    """
    print("APLICANDO UPSCALING CON REAL-ESRGAN (API REAL)...")
    
    # Verificar si existe el token antes de intentar nada
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("⚠️ Token de Replicate no encontrado. Usando modo offline.")
        # No mostramos error crítico en UI para no asustar, solo fallback silencioso o aviso warning
        st.warning("⚠️ No se configuró el token de Replicate. Usando escalado tradicional.")
        width, height = image.size
        return image.resize((width * 4, height * 4), Image.Resampling.LANCZOS)

    try:
        # 1. Convertir imagen a bytes para enviarla a la nube
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # 2. Llamada al modelo en Replicate
        output = replicate.run(
            "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73ab415c7298a3b9d8503",
            input={
                "image": BytesIO(img_byte_arr),
                "scale": 4,
                "face_enhance": True  # ¡IMPORTANTE! Esto arregla las caras
            }
        )
        
        # 3. Descargar la imagen generada
        # La API puede devolver string o lista, manejamos ambos casos
        image_url = output if isinstance(output, str) else output[0]
        response = requests.get(image_url)
        upscaled_img = Image.open(BytesIO(response.content))
        
        return upscaled_img

    except Exception as e:
        # --- MANEJO DE ERRORES ---
        # Si falla (ej. se acabó el crédito, error de servidor), mostramos el error en rojo
        error_msg = str(e)
        st.error(f"⚠️ Error en Super-Resolución (Replicate): {error_msg}")
        print(f"ERROR REPLICATE: {error_msg}")
        
        # FALLBACK: Devolvemos imagen reescalada manualmente para que la app continúe
        scale_factor = 4
        width, height = image.size
        return image.resize(
            (width * scale_factor, height * scale_factor), 
            Image.Resampling.LANCZOS
        )