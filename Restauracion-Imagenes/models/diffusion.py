# models/diffusion.py
import numpy as np
from PIL import Image
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import streamlit as st  # <--- IMPORTANTE

load_dotenv()

def process_image(image, prompt, intensity, control_maps, face_embedding, person_mask):
    """
    Procesa la imagen usando Stable Diffusion.
    """
    # Usamos SD 1.5 que es fiable para mejoras generales
    repo_id = "runwayml/stable-diffusion-v1-5"
    
    # Verificación rápida del token
    token = os.getenv("HF_TOKEN")
    if not token:
        st.error("❌ ERROR CRÍTICO: No se encontró HF_TOKEN en el archivo .env")
        return image
    
    client = InferenceClient(token=token)
    
    print(f"Iniciando IA con Intensidad: {intensity}")

    try:
        # Asegurar modo RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Llamada a la API
        output_image = client.image_to_image(
            model=repo_id,
            prompt=prompt + ", high resolution, realistic, 8k, sharp focus",
            negative_prompt="blur, low quality, cartoon, bad anatomy, distorted face, glitch, artifacts",
            image=image,
            strength=intensity, # Si esto es muy bajo (<0.3) apenas se notará el cambio
            guidance_scale=7.5
        )
        
        return output_image

    except Exception as e:
        # --- AQUÍ ESTÁ LA CLAVE ---
        # Mostramos el error rojo en la pantalla de la app
        st.error(f"⚠️ Error en Stable Diffusion: {str(e)}")
        print(f"ERROR REAL: {e}")
        return image