# 📸 Restauración Inteligente de Imágenes (Real-ESRGAN + Stable Diffusion)

> Sistema avanzado de recuperación y mejora de imágenes que combina Super-Resolución clásica con IA Generativa para restaurar fotos antiguas o de baja calidad.

## 📝 Descripción

Este proyecto es un MVP (Producto Mínimo Viable) desarrollado para la materia **Procesamiento Digital de Imágenes**. 

El sistema resuelve el problema de la **degradación visual** en fotografías digitales (baja resolución, ruido, falta de nitidez). A diferencia de los filtros tradicionales, utiliza un pipeline híbrido: primero restaura la estructura geométrica con **Real-ESRGAN** y luego alucina detalles realistas perdidos utilizando **Stable Diffusion** guiado por segmentación (DETR), asegurando que la mejora no altere la identidad de los sujetos.

## 👤 User Persona

El sistema fue diseñado pensando en:

* **Nombre:** Ana "La Archivista"
* **Perfil:** 45 años, entusiasta de la genealogía familiar.
* **Problema:** Heredó cientos de fotos digitales de los años 2000 y escaneos viejos que se ven pixelados y "sucios" en las pantallas modernas 4K.
* **Necesidad:** Una herramienta simple (sin código) que mejore la calidad para imprimir álbumes, sin que las caras de sus familiares parezcan "de plástico" o deformes.
* **Solución:** Una interfaz web donde sube la foto, ajusta qué tanto quiere que intervenga la IA, y descarga el resultado listo para imprimir.

## 🚀 Demo

**[Ver Video Demo en YouTube/Loom](https://drive.google.com/file/d/1dYunc6ojcnbWxZ1YYQZB0DxrOTP1blsB/view?usp=drive_link)**

## ⚙️ Características Técnicas

1.  **Restauración Estructural (Real-ESRGAN):** Upscaling x2/x4 eliminando artefactos de compresión JPG.
2.  **Reconstrucción Generativa (Stable Diffusion 1.5):** Inferencia imagen-a-imagen para agregar texturas de alta frecuencia (pelo, madera, tela).
3.  **Segmentación Inteligente (DETR):** Detecta personas para aplicar máscaras de protección (evitando deformaciones en rostros).
4.  **Análisis de Calidad:** Cálculo automático de métricas (Similitud de bordes Canny y CLIP Score) para validar la mejora objetivamente.
5.  **Comparación A/B:** Visor interactivo para comparar el antes y el después.

## 🛠️ Tecnologías Utilizadas

* **Frontend:** Streamlit
* **Core IA:**
    * `Real-ESRGAN` (Local, Pytorch implementation via `basicsr`)
    * `Stable Diffusion v1.5` (via Hugging Face Inference API)
    * `DETR` (Facebook Detection Transformer)
* **Procesamiento:** OpenCV, PIL, NumPy.
* **Infraestructura:** Python 3.10 (Requerido por compatibilidad con Torchvision).

## 💻 Instalación y Configuración Local

Este proyecto requiere una configuración específica debido a la incompatibilidad entre librerías modernas de PyTorch y módulos legacy (`basicsr`). Siga estos pasos al pie de la letra.

### Prerrequisitos
* **Python 3.10** (Obligatorio. Versiones 3.11 o 3.12 causarán errores).
* **Git** instalado.
* Una cuenta en Hugging Face (para el token de API).

### Pasos

1.  **Clonar el repositorio:**
    ```bash
    git clone (https://github.com/Josnel-ME/Tecnicas-de-procesamiento-digital-de-imagenes.git)
    cd Restauracion-Imagenes
    ```

2.  **Crear un entorno virtual con Python 3.10:**
    Es vital forzar el uso de Python 3.10. En Windows:
    ```bash
    # Opción A: Si tienes el Python Launcher
    py -3.10 -m venv venv

    # Opción B: Ruta directa (ejemplo)
    C:\Python310\python.exe -m venv venv
    ```

3.  **Activar el entorno:**
    ```bash
    # Windows
    .\venv\Scripts\activate
    
    # Linux/Mac
    source venv/bin/activate
    ```

4.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **🔧 CORRECCIÓN MANUAL (CRÍTICO):**
    La librería `basicsr` tiene una incompatibilidad con las versiones nuevas de `torchvision`. Debe editar un archivo manualmente para que el proyecto funcione:

    * Navegue a: `venv/Lib/site-packages/basicsr/data/degradations.py`
    * Abra el archivo y busque la **línea 8**:
        ```python
        from torchvision.transforms.functional_tensor import rgb_to_grayscale
        ```
    * **Edítela** para borrar la palabra `_tensor`. Debe quedar así:
        ```python
        from torchvision.transforms.functional import rgb_to_grayscale
        ```
    * Guarde el archivo.

6.  **Configurar Variables de Entorno:**
    Cree un archivo llamado `.env` en la carpeta raíz del proyecto y agregue su token:
    ```env
    HF_TOKEN=hf_TuTokenDeHuggingFaceAqui
    ```

7.  **Ejecutar la aplicación:**
    ```bash
    streamlit run app.py
    ```