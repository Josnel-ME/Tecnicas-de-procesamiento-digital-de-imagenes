# Aplicación de Restauración con RealESRGAN + Stable Diffusion
# Archivo: app.py

import streamlit as st
from PIL import Image
import os
from dotenv import load_dotenv

# Modelos propios
from models.diffusion import process_image
from models.analysis import analyze_images
from models.segmentation import get_person_mask
from models.realesrgan_upscale import upscale_realesrgan

# Utilidades
from utils.image_utils import (
    extract_control_maps,
    extract_face_embedding,
    upscale_image,
    adjust_brightness,
    adjust_sharpness
)

load_dotenv()

st.set_page_config(page_title="Restauración Avanzada", layout="wide")
st.title("📸 Restauración Inteligente de Imágenes (RealESRGAN + Stable Diffusion)")

# Sidebar
with st.sidebar:
    st.header("Configuración IA")
    prompt = st.text_input(
        "Prompt de mejora",
        "High quality, realistic, 8k, sharp focus, clean details"
    )
    intensity = st.slider(
        "Intensidad Stable Diffusion (Denoising)",
        0.1, 1.0, 0.45,
        help="Valores bajos conservan más la imagen original."
    )

    st.subheader("Preprocesamiento clásico")
    brightness_factor = st.slider("Brillo", 0.5, 1.5, 1.0, 0.05)
    sharpness_factor = st.slider("Nitidez", 0.0, 3.0, 1.5, 0.1)

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img_original = Image.open(uploaded_file).convert("RGB")

    # Pre-procesamiento
    img_adjusted = adjust_brightness(img_original, brightness_factor)
    img_clean = adjust_sharpness(img_adjusted, sharpness_factor)

    st.subheader("1️⃣ Original vs Imagen Preprocesada")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_original, caption="Original", use_column_width=True)
    with col2:
        st.image(img_clean, caption="Preprocesada", use_column_width=True)

    if st.button("🚀 Restaurar Imagen"):
        with st.status("Procesando...", expanded=True) as status:

            st.write("🔧 **RealESRGAN: Restauración real antes de la IA**")
            restored_img = upscale_realesrgan(img_clean, scale=2)

            st.write("🔍 Segmentando la imagen... (DETR)")
            person_mask = get_person_mask(restored_img)

            control_maps = extract_control_maps(restored_img)
            face_embedding = extract_face_embedding(restored_img)

            st.write("🎨 **Stable Diffusion: reconstruyendo detalles**")
            sd_img = process_image(
                restored_img,
                prompt,
                intensity,
                control_maps,
                face_embedding,
                person_mask
            )

            st.write("📐 Upscaling final (RealESRGAN local)")
            # Usamos el mismo modelo local que en el paso 1
            final_img = upscale_realesrgan(sd_img, scale=2)

            st.write("📊 Analizando calidad...")
            analysis = analyze_images(restored_img, final_img, prompt)

            status.update(label="✔ Restauración completa", state="complete", expanded=False)

        st.header("Resultado Final")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_original, caption="Original", use_column_width=True)
        with col2:
            st.image(final_img, caption="Restaurada", use_column_width=True)

        # Panel de métricas
        st.subheader("📊 Análisis de Calidad (Visión por Computadora)")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.info(f"**Estructura (Canny):** {analysis['estructura']}")

        with col_b:
            clip_score = analysis.get("clip_score")
            if clip_score is None:
                st.warning("CLIP Score no disponible")
            else:
                st.success(f"CLIP Score: {clip_score:.4f}")

        with col_c:
            st.info(f"**Iluminación:** {analysis['iluminacion']}")

        st.download_button(
            "Descargar Imagen Restaurada",
            data=final_img.tobytes(),
            file_name="restaurada.png",
            mime="image/png"
        )
