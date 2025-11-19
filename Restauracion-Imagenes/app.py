
import streamlit as st
from PIL import Image
import os
from dotenv import load_dotenv

# Importar modelos y utilidades
from models.diffusion import process_image
from models.analysis import analyze_images
from models.segmentation import get_person_mask 
from utils.image_utils import (
    extract_control_maps,
    extract_face_embedding,
    upscale_image,
    adjust_brightness,
    adjust_sharpness
)

# Cargar variables de entorno
load_dotenv()

# Configuración de la página
st.set_page_config(page_title="Restauración con IA", layout="wide")

def show_images_side_by_side(img_orig, img_proc, caption_orig="Original", caption_proc="Procesada"):
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_orig, caption=caption_orig, use_column_width=True)
    with col2:
        st.image(img_proc, caption=caption_proc, use_column_width=True)

# --- Inicio de la App ---
st.title('✨ Sistema de Restauración y Mejora de Imágenes')
st.markdown("""
Esta herramienta permite restaurar fotos antiguas o de baja calidad utilizando 
**Modelos de Difusión** y verificando el resultado con **Visión por Computadora**.
""")

# Sidebar
with st.sidebar:
    st.header("Configuración")
    prompt = st.text_input('Prompt de mejora', 'High quality, realistic, 8k, sharp focus, professional photography')
    intensity = st.slider('Intensidad de la IA (Denoising Strength)', 0.1, 1.0, 0.4, help="Valores bajos conservan más la original. Valores altos inventan más detalles.")
    
    st.divider()
    st.subheader("Pre-procesamiento (PDI)")
    brightness_factor = st.slider("Brillo", 0.5, 1.5, 1.0, 0.05)
    sharpness_factor = st.slider("Nitidez", 0.0, 3.0, 1.5, 0.1)

uploaded_file = st.file_uploader('Sube una imagen', type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # 1. Cargar imagen
    img_original = Image.open(uploaded_file).convert("RGB")
    
    # 2. Pre-procesamiento (PDI clásico en tiempo real)
    img_adjusted = adjust_brightness(img_original, brightness_factor)
    img_clean = adjust_sharpness(img_adjusted, sharpness_factor)

    # Mostrar estado actual
    st.subheader("1. Imagen Original vs Pre-procesada")
    show_images_side_by_side(img_original, img_clean, "Original", "Entrada para la IA (Brillo/Nitidez ajustados)")

    # Botón para ejecutar la IA (costoso, por eso usamos botón)
    if st.button("🚀 Procesar con Inteligencia Artificial"):
        
        with st.status("Procesando imagen...", expanded=True) as status:
            
            # 3. Segmentación (Análisis Visual)
            st.write("🔍 Detectando personas y analizando estructura...")
            person_mask = get_person_mask(img_clean)
            # Estos son placeholders requeridos por tu lógica actual
            control_maps = extract_control_maps(img_clean)
            face_embedding = extract_face_embedding(img_clean)

            # 4. Difusión (Stable Diffusion)
            st.write("🎨 Generando detalles con Stable Diffusion...")
            enhanced_img_local = process_image(
                img_clean, prompt, intensity, control_maps, face_embedding, person_mask
            )

            # 5. Super-Resolución (Real-ESRGAN)
            st.write("📐 Aumentando resolución (Upscaling)...")
            final_enhanced_img = upscale_image(enhanced_img_local)
            
            # 6. Análisis final
            st.write("📊 Calculando métricas de calidad...")
            analysis = analyze_images(img_clean, final_enhanced_img, prompt)
            
            status.update(label="¡Procesamiento completado!", state="complete", expanded=False)

        # 7. Resultados Finales
        st.markdown("---")
        st.header("Resultado Final")
        show_images_side_by_side(img_original, final_enhanced_img, "Original", "Restaurada con IA")
        
        # 8. Panel de Análisis
        st.markdown("### 📊 Análisis de Calidad (Visión por Computadora)")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.info(f"**Estructura (Canny):**\n{analysis['estructura']}")
        with col_b:
            st.success(f"**Similitud Semántica (CLIP):**\n{analysis.get('clip_score', 'N/A'):.4f}")
        with col_c:
            st.warning(f"**Análisis de Luz:**\n{analysis['iluminacion']}")
            
        # Opción de descarga
        st.download_button("Descargar Imagen Procesada", data=final_enhanced_img.tobytes(), file_name="restaurada.png", mime="image/png")