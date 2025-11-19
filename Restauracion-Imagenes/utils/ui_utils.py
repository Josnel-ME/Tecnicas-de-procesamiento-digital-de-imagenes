# utils/ui_utils.py
import streamlit as st

def show_images_side_by_side(original_img, enhanced_img):
    """
    Muestra las imágenes original y mejorada lado a lado en Streamlit.
    """
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_img, caption='Imagen Original')
    with col2:
        st.image(enhanced_img, caption='Imagen Mejorada')
