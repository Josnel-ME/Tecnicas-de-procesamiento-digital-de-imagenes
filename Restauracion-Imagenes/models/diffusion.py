# models/diffusion.py

from PIL import ImageEnhance, Image
import numpy as np
import cv2

# Asegúrate de que esta función tome la 'person_mask' como argumento.
def process_image(image, prompt, intensity, control_maps, face_embedding, person_mask):
    """
    Procesa la imagen aplicando mejoras en la región segmentada de la persona 
    (obtenida de un modelo de segmentación) y mezcla el resultado suavemente.

    Args:
        image: Imagen original (PIL)
        prompt: Prompt de mejora
        intensity: Intensidad de la mejora (0.1 a 1.0)
        control_maps: Diccionario con mapas de control (no usado en la simulación)
        face_embedding: Embedding facial (no usado en la simulación)
        person_mask: Máscara binaria (0 o 255) de la persona (np.ndarray)
    
    Returns:
        Imagen mejorada (PIL)
    """
    img_np = np.array(image)
    
    # --- 1. Usar la máscara precisa ---
    mask = person_mask # Máscara obtenida de models/segmentation.py (DETR/SAM)
    
    # --- 2. Suavizar la máscara (feathering) para una transición suave ---
    # Usamos un kernel de desenfoque proporcional al 5% del ancho de la imagen
    kernel_size = int(img_np.shape[1] * 0.05)
    # Aseguramos que el tamaño del kernel sea impar (requerido por cv2.GaussianBlur)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1 
    
    # Desenfoque gaussiano para crear la máscara de blending (0.0 a 1.0)
    mask_blur = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    mask_blur = mask_blur / 255.0  # Normalizar a [0, 1]

    # --- 3. Aplicar el Enhancement (CORE del Track 2) ---
    enhanced_img = image.copy()
    
    # NOTA: En un proyecto real, la siguiente línea sería la llamada al modelo 
    # de Stable Diffusion img2img para aplicar el prompt y la iluminación/color.
    
    # SIMULACIÓN (Usando PDI de Transformaciones de Intensidad):
    enhancer = ImageEnhance.Brightness(enhanced_img)
    # Ajuste de brillo: 1.0 (original) + intensidad (0.1 a 1.0) * factor
    enhanced_img = enhancer.enhance(1.0 + intensity * 0.5) 
    
    enhanced_np = np.array(enhanced_img)
    
    # --- 4. Mezclar la región mejorada con el fondo original ---
    
    # Expandir la máscara borrosa a 3 canales para la multiplicación
    mask_3d = mask_blur[..., None]
    
    # result = (Región Mejorada * Máscara) + (Fondo Original * Complemento de Máscara)
    result = (enhanced_np * mask_3d + img_np * (1 - mask_3d)).astype(np.uint8)
    
    return Image.fromarray(result)