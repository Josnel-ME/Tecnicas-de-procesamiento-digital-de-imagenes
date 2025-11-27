#  🌟  Detector de Landmarks Faciales

## Introducción

Este repositorio presenta una **Aplicación Web para la Detección de Landmarks Faciales** utilizando librerías de vanguardia en Visión por Computadora. El objetivo principal es ofrecer una herramienta interactiva y en tiempo real capaz de identificar y mapear **478 puntos clave** (landmarks) en rostros humanos.

El sistema está diseñado para demostrar las capacidades de la librería **MediaPipe** en la detección de características faciales, proporcionando una interfaz sencilla e interactiva construida con **Streamlit**.

---

## Contenido

Esta carpeta contiene todos los archivos necesarios para la ejecución y demostración de la aplicación:

| Archivo | Tipo | Descripción |
| :--- | :--- | :--- |
| `app.py` | Script Python | **Código principal** de la aplicación web Streamlit, que implementa la lógica de detección de MediaPipe. |
| `requirements.txt` | Configuración | Lista de librerías y versiones necesarias para instalar las dependencias del proyecto. |

---

## Resumen del Sistema y Características

El sistema es una aplicación web interactiva y en tiempo real que aplica algoritmos de detección avanzada para el análisis facial.

### 🌟 Características Principales

* **Detección de 478 Landmarks Faciales:** Identifica un conjunto extenso de puntos clave para un mapeo detallado del rostro.
* **Interfaz Web Interactiva:** Creada con Streamlit para una experiencia de usuario fluida.
* **Procesamiento en Tiempo Real:** Capacidad para procesar *frames* de video o imágenes con baja latencia.
* **Visualización Antes/Después:** Permite comparar la imagen o *stream* original con la superposición de los landmarks detectados.


### Librerías Utilizadas

| Librería | Propósito |
| :--- | :--- |
| **MediaPipe** | Herramienta principal para el **reconocimiento y detección de los 478 landmarks faciales**. |
| **OpenCV** (cv2) | Utilizada para tareas fundamentales de **procesamiento de imágenes** y manipulación de *frames*. |
| **Streamlit** | Framework principal para la construcción de la **interfaz web interactiva** y la demostración en vivo. |
| **Python 3.11+** | Versión mínima requerida del lenguaje de programación. |

---

## 💻 Instalación y Ejecución Local

Para utilizar y probar esta aplicación en tu máquina local, sigue los siguientes pasos:

1.  **Clonar el repositorio:**

    ```bash
    git clone https://github.com/RaquelMallqui1/Tecnicas-de-Procesamiento-Digital-de-Imagenes.git
    cd facial-landmarks-app
    ```

2.  **Crear y activar el entorno virtual:**

    ```bash
    # Crear entorno virtual
    python -m venv venv

    # Activar en Windows
    venv\Scripts\activate

    # Activar en Linux/Mac
    source venv/bin/activate
    ```

3.  **Instalar dependencias:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecutar la aplicación Streamlit:**

    ```bash
    streamlit run app.py
    ```

Esto abrirá la aplicación web en tu navegador predeterminado (normalmente en `http://localhost:8501`).

---

## 📧 Contacto y Licencia

### Contacto

Para consultas o sugerencias relacionadas con la implementación del detector de landmarks faciales:

* **Nombre:** Josnel Malqui Espinoza
* **Email:** 93854948@ifts24.edu.ar

### Licencia

Este proyecto está disponible para su **Uso Educativo y de Demostración**. Se prohíbe su uso comercial sin consentimiento explícito.
