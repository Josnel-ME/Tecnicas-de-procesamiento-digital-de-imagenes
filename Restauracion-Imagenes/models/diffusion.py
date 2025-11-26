# models/diffusion.py
import numpy as np
from PIL import Image
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import io
import streamlit as st  

load_dotenv()

def process_image(image, prompt, intensity, control_maps, face_embedding, person_mask):
    repo_id = "runwayml/stable-diffusion-v1-5"
    token = os.getenv("HF_TOKEN")

    if not token:
        st.error("❌ No se encontró HF_TOKEN en el archivo .env")
        return image

    client = InferenceClient(repo_id, token=token)

    try:
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convertir PIL -> bytes
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        result = client.image_to_image(
            image=img_bytes,
            prompt=prompt,
            strength=float(intensity),
            negative_prompt="blur, low quality, artifacts, distorted",
            guidance_scale=7.5
        )

        return Image.open(io.BytesIO(result))

    except Exception as e:
        st.error(f"⚠️ Error en Stable Diffusion: {str(e)}")
        return image