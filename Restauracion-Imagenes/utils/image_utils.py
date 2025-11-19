# utils/image_utils.py

def extract_control_maps(image):
    """
    Extrae mapas de control (Canny, Depth) de la imagen.
    Returns:
        Dict con mapas de control
    """
    # TODO: Implementar extracción de mapas
    return {'canny': None, 'depth': None}

def extract_face_embedding(image):
    """
    Extrae el embedding facial usando IP-Adapter/InstantID.
    Returns:
        Embedding facial
    """
    # TODO: Implementar extracción de embedding facial
    return None

# --- Procesamiento digital clásico ---
import cv2
import numpy as np
from PIL import Image

def equalize_histogram(image):
    """
    Aplica ecualización de histograma para mejorar el contraste.
    Soporta imágenes en escala de grises y color.
    """
    img = np.array(image)
    if len(img.shape) == 2:
        # Escala de grises
        eq = cv2.equalizeHist(img)
    else:
        # Color: ecualiza cada canal
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(eq)

def sharpen_image(image):
    """
    Aplica filtro de nitidez (Unsharp Mask).
    """
    img = np.array(image)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharp = cv2.filter2D(img, -1, kernel)
    return Image.fromarray(sharp)

def reduce_artifacts(image):
    """
    Reduce artefactos de compresión usando filtro bilateral.
    """
    img = np.array(image)
    smooth = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    return Image.fromarray(smooth)
