# models/realesrgan_upscale.py

import numpy as np
from PIL import Image
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet  # <--- IMPORTACIÓN CLAVE
import os
import requests

def upscale_realesrgan(img: Image.Image, scale=2, model_path="RealESRGAN_x2plus.pth"):
    """
    Upscaling con Real-ESRGAN oficial.
    Define la arquitectura RRDBNet antes de cargar los pesos.
    """
    
    # 1. Descarga automática del modelo si no existe
    if not os.path.exists(model_path):
        print(f"⬇️ Descargando {model_path}...")
        # URL oficial para x2plus
        url = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/{model_path}"
        try:
            response = requests.get(url)
            response.raise_for_status() # Verificar si la descarga fue exitosa
            with open(model_path, "wb") as f:
                f.write(response.content)
            print("✅ Descarga completa.")
        except Exception as e:
            print(f"❌ Error descargando modelo: {e}")
            return img # Retornar original si falla la descarga

    # 2. DEFINIR LA ARQUITECTURA (El "Esqueleto" del modelo)
    # Estos parámetros son específicos para el modelo x2plus y x4plus
    model = RRDBNet(
        num_in_ch=3, 
        num_out_ch=3, 
        num_feat=64, 
        num_block=23, 
        num_grow_ch=32, 
        scale=scale
    )

    # 3. Preparar imagen
    img_np = np.array(img)
    if img_np.ndim == 2:
        img_np = np.stack([img_np]*3, axis=-1)
    if img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 4. Crear el upsampler pasando el modelo definido
    try:
        upsampler = RealESRGANer(
            scale=scale,
            model_path=model_path,
            model=model,       
            dni_weight=None,
            device=device
        )

        output, _ = upsampler.enhance(img_np, outscale=scale)
        return Image.fromarray(output)
        
    except Exception as e:
        print(f"Error crítico en RealESRGAN: {e}")
        return img