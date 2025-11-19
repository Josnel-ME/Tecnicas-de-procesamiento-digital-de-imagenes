import streamlit as st
from models.diffusion import process_image
from models.analysis import analyze_images
from models.segmentation import get_person_mask

from utils.image_utils import (
    extract_control_maps,
    extract_face_embedding,
    equalize_histogram,
    sharpen_image,
    reduce_artifacts
)
from utils.ui_utils import show_images_side_by_side
import yaml

from PIL import Image

st.title('Restauración y Mejora de Imágenes')

uploaded_file = st.file_uploader('Sube una imagen', type=['jpg', 'jpeg', 'png'])
prompt = st.text_input('Prompt de mejora', 'Iluminación cinematográfica, colores vivos, alta definición')
intensity = st.slider('Intensidad de la mejora', 0.1, 1.0, 0.5)

if uploaded_file:
    # Convertir archivo subido a imagen PIL
    img = Image.open(uploaded_file)
    # Aplica procesamiento digital clásico
    img_eq = equalize_histogram(img)
    img_sharp = sharpen_image(img_eq)
    img_clean = reduce_artifacts(img_sharp)

    person_mask = get_person_mask(img_clean)

    control_maps = extract_control_maps(img_clean)
    face_embedding = extract_face_embedding(img_clean)

    # Difusión
    enhanced_img = process_image(
        img_clean, prompt, intensity, control_maps, face_embedding, person_mask
    )

    # Análisis visual
    analysis = analyze_images(img_clean, enhanced_img, prompt)

    # Visualización
    show_images_side_by_side(img_clean, enhanced_img)
    st.write('Análisis visual:', analysis)
