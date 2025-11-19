# models/segmentation.py
# models/segmentation.py

from transformers import DetrForSegmentation, DetrImageProcessor
from PIL import Image
import torch
import numpy as np
import cv2
import logging

# Silenciar advertencias técnicas
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# Cargar modelos
try:
    print("Cargando modelos DETR...")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50-panoptic")
    model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")
    print("Modelos cargados.")
except Exception as e:
    print(f"Error cargando modelos DETR: {e}")

def get_person_mask(image: Image.Image) -> np.ndarray:
    """
    Genera una máscara binaria para la persona usando DETR.
    SOLUCIÓN FINAL: Usa una lista de tuplas estándar de Python para target_sizes.
    """
    # 1. Asegurar formato RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 2. Preparar inputs
    inputs = processor(images=image, return_tensors="pt")
    
    # 3. Mover a GPU si está disponible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 4. Inferencia
    with torch.no_grad():
        outputs = model(**inputs)

    # --- CORRECCIÓN CRÍTICA ---
    # En lugar de usar torch.tensor, usamos una lista de tuplas simple.
    # Esto evita que la librería interna cree una "lista de tensores" que rompe PyTorch.
    # image.size es (Ancho, Alto), necesitamos (Alto, Ancho)
    height, width = image.size[1], image.size[0]
    target_sizes = [(height, width)]
    # --------------------------

    # 5. Post-procesamiento
    try:
        results = processor.post_process_panoptic_segmentation(
            outputs, 
            target_sizes=target_sizes
        )[0]
    except Exception as e:
        print(f"Error en post-procesamiento: {e}")
        # Fallback de emergencia: devolver máscara vacía para no romper la app
        return np.zeros((height, width), dtype=np.uint8)

    # 6. Generar máscara binaria
    mask_array = results["segmentation"].cpu().numpy()
    segments_info = results["segments_info"]

    person_id = 1 
    person_mask = np.zeros_like(mask_array, dtype=np.uint8)
    found_person = False

    for segment in segments_info:
        label = segment.get('label_id', segment.get('category_id'))
        
        if label == person_id:
            segment_id = segment['id']
            current_mask = (mask_array == segment_id).astype(np.uint8) * 255
            person_mask = cv2.bitwise_or(person_mask, current_mask)
            found_person = True

    if not found_person:
        print("⚠️ Aviso: No se detectó persona en la imagen.")

    return person_mask
