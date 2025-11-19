# 🎓 Proyecto Integrador: Procesamiento de Imágenes para Mejora de Calidad

Este proyecto tiene como objetivo demostrar la aplicación de diversas **técnicas de procesamiento de imágenes** para mejorar la calidad y la legibilidad de documentos de texto con defectos comunes como rotación, sombras y baja calidad.

---

## 🎯 Contenido y Objetivo

El proyecto se centra en el procesamiento de tres imágenes de documentos de texto, utilizando el *notebook* **`proyecto_integrador.ipynb`** para documentar el proceso paso a paso.

| Archivo | Problema Específico | Solución Aplicada |
| :--- | :--- | :--- |
| **`imagen_buena.jpg`** | Imagen de referencia | Utilizada para fines comparativos. |
| **`imagen_rotada.jpg`** | Rotación del documento | Requiere enderezamiento para ser legible. |
| **`imagen_con_sombras.jpg`** | Sombras y Baja Calidad | Requiere segmentación de texto para visibilidad. |

---

## ⚙️ Técnicas Aplicadas

Para abordar los problemas de cada imagen, se utilizaron las siguientes técnicas de procesamiento de imágenes:

* **Transformación Geométrica:** Aplicación de **rotación** precisa a la imagen rotada para alinear correctamente el documento.
* **Umbralización (Thresholding):** Uso de la **umbralización adaptativa** para segmentar el texto del fondo en imágenes con sombras, eliminando las variaciones de iluminación.
* **Mejora de Calidad / Reducción de Ruido:** Aplicación de **filtros de mediana** para reducir el ruido y mejorar la nitidez en la imagen de baja calidad.

### Resultados

En el *notebook*, se muestra el proceso paso a paso y los resultados de cada técnica. Al final, se puede comparar la imagen original con la imagen procesada para ver la mejora significativa en la calidad y la legibilidad del documento.

---

## 💻 Ejecución del Proyecto

Para ejecutar este proyecto, simplemente abre el *notebook* **`proyecto_integrador.ipynb`** en **Google Colab** o **Jupyter Notebook**.

### Dependencias

Asegúrate de tener las librerías necesarias instaladas en tu entorno virtual. Las dependencias clave son:

```bash
pip install opencv-python numpy matplotlib
