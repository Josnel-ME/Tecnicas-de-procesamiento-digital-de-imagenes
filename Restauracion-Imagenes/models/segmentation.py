# models/segmentation.py

from transformers import DetrForSegmentation, DetrImageProcessor
from PIL import Image
import torch
import numpy as np
import cv2

# Cargar el modelo de segmentación (Panoptic Segmentation)
# Mantenemos las cargas del modelo fuera de la función para mayor eficiencia
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")
model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")

def get_person_mask(image: Image.Image) -> np.ndarray:
    """
    Genera una máscara binaria para la persona usando DETR (Hugging Face).
    
    Returns:
        np.ndarray: Máscara binaria (0 o 255) del tamaño de la imagen.
    """
    inputs = processor(images=image, return_tensors="pt")

    # Mover el input al dispositivo correcto si usas GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # 1. Post-procesamiento para obtener la segmentación panóptica
    panoptic_output = processor.post_process_panoptic_segmentation(
        outputs, 
        target_sizes=[image.size[::-1]] # (altura, ancho)
    )[0]
    
    # ID de la categoría "persona" en el dataset COCO (común para este modelo)
    person_id = 1 
    
    # 2. Extraer el mapa de segmentación (donde cada píxel tiene un ID de segmento)
    mask_array = np.array(panoptic_output['segmentation'])
    
    # 3. Inicializar la máscara final
    person_mask = np.zeros_like(mask_array, dtype=np.uint8)
    
    # 4. Iterar sobre la información de los segmentos
    for segment in panoptic_output['segments_info']:
        
        # --- CORRECCIÓN CLAVE: Usar 'label_id' o 'id' ---
        # El campo que contiene la ID de la clase de COCO (1=persona) es a menudo 'label_id' o 'id'.
        # Usaremos 'label_id' que es más estándar en el output post-procesado.
        
        # Verificar si la clave 'label_id' existe y si corresponde a la persona
        if 'label_id' in segment and segment['label_id'] == person_id:
            
            # Obtener el ID del segmento específico del mapa de segmentación
            segment_id = segment['id']
            
            # Crear una máscara binaria para este segmento específico
            mask_segment = (mask_array == segment_id).astype(np.uint8) * 255
            
            # Sumar esta máscara al resultado final (bitwise_or)
            person_mask = cv2.bitwise_or(person_mask, mask_segment)
        
        elif 'category_id' in segment and segment['category_id'] == person_id:
             # Este es el caso que originalmente buscabas, lo mantenemos como fallback/check
            segment_id = segment['id']
            mask_segment = (mask_array == segment_id).astype(np.uint8) * 255
            person_mask = cv2.bitwise_or(person_mask, mask_segment)

    # El post-procesamiento de DETR ya debería retornar la máscara del tamaño correcto
    # Retornamos la máscara binaria (0 o 255)
    return person_mask