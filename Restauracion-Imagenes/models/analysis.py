# models/analysis.py

import numpy as np
from PIL import Image
import cv2
try:
    import torch
    import clip
except ImportError:
    clip = None

def analyze_images(original_img, enhanced_img, prompt):
    """
    Analiza visualmente la imagen original y la mejorada usando histogramas y CLIP Score.
    Args:
        original_img: Imagen original (PIL)
        enhanced_img: Imagen mejorada (PIL)
        prompt: Prompt de mejora
    Returns:
        Dict con resultados del análisis
    """
    # --- Iluminación ---
    def get_brightness_stats(img):
        arr = np.array(img.convert('L'))
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        return mean, std
    mean_orig, std_orig = get_brightness_stats(original_img)
    mean_enh, std_enh = get_brightness_stats(enhanced_img)
    diff_mean = mean_enh - mean_orig
    diff_std = std_enh - std_orig
    iluminacion = (
        f"Brillo original: {mean_orig:.2f} (std: {std_orig:.2f}), "
        f"mejorada: {mean_enh:.2f} (std: {std_enh:.2f}). "
        f"Diferencia de brillo: {diff_mean:.2f}, diferencia de contraste: {diff_std:.2f}"
    )

    # --- CLIP Score ---
    clip_score = None
    if clip is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        image_input = preprocess(enhanced_img).unsqueeze(0).to(device)
        text_input = clip.tokenize([prompt]).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).item()
            clip_score = similarity

    # --- Estructura (bordes Canny) ---
    def get_canny_edges(img: Image.Image, target_size: tuple = None):
        """Calcula bordes Canny. Reescala si se especifica target_size."""
        
        # Si la imagen mejorada es más grande, la reescalamos para el análisis
        if target_size and img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            
        arr = np.array(img.convert('L'))
        edges = cv2.Canny(arr, 100, 200)
        return edges


# Usar el tamaño de la imagen original como referencia
    target_size_orig = original_img.size
    
    edges_orig = get_canny_edges(original_img, target_size=target_size_orig)
    # Reescalar la imagen mejorada a la escala de la original ANTES de calcular Canny
    edges_enh = get_canny_edges(enhanced_img, target_size=target_size_orig) 
    
    # Ambos arrays ahora tienen el mismo tamaño, ¡el broadcasting funcionará!
    intersection = np.logical_and(edges_orig, edges_enh)
    union = np.logical_or(edges_orig, edges_enh)
    
    if np.sum(union) == 0:
        estructura_score = 1.0
    else:
        estructura_score = np.sum(intersection) / np.sum(union)
    
    estructura = f"Similitud de bordes: {estructura_score:.2f}"

    return {
        'iluminacion': iluminacion,
        'identidad': 'Pendiente', # Requiere InstantID/IP-Adapter
        'estructura': estructura,
        'artefactos': 'Pendiente', # Requiere métrica de artefactos
        'clip_score': clip_score
    }


