# 🖼️ Proyecto Integrador: Procesamiento de Imágenes para Mejora de Calidad

Este proyecto tiene como objetivo demostrar la aplicación de diversas **técnicas de procesamiento de imágenes** para mejorar la calidad y la legibilidad de documentos de texto con defectos comunes como **rotación**, **sombras** y **baja calidad/ruido**.

---

## 🎯 Contenido del Proyecto

El proyecto se centra en el procesamiento de tres imágenes de documentos de texto, cada una con un problema específico que requiere una solución diferente:

| Archivo de Imagen | Problema Detectado | Descripción |
| :--- | :--- | :--- |
| `imagen_buena.jpg` | **Referencia** | Imagen de buena calidad utilizada para fines comparativos. |
| `imagen_rotada.jpg` | **Rotación** | Imagen que se encuentra rotada y necesita ser enderezada para ser legible. |
| `imagen_con_sombras.jpg` | **Sombras/Baja Iluminación** | Imagen con sombras que oscurecen el texto, afectando su visibilidad. |
| **_Nota:_** La imagen de baja calidad se procesa dentro del contexto de la imagen con sombras, o se asume que una de las mencionadas presenta el defecto de calidad. |

---

## 🛠️ Técnicas de Procesamiento Aplicadas

Para abordar los problemas de cada imagen, se utilizaron las siguientes técnicas de procesamiento de imágenes:

* **Transformación Geométrica:** Se aplicó una **rotación** precisa a `imagen_rotada.jpg` para alinear correctamente el documento y facilitar su lectura.
* **Umbralización (Thresholding):** Se usó la **umbralización adaptativa** en `imagen_con_sombras.jpg`. Esta técnica es crucial para segmentar el texto del fondo, eliminando eficazmente las variaciones de iluminación causadas por las sombras.
* **Mejora de Calidad / Reducción de Ruido:** Se aplicaron **filtros** (como el filtro de mediana) para reducir el ruido y mejorar la nitidez en la imagen de baja calidad (o la imagen con sombras después de la umbralización).

---

## 📈 Resultados

El _notebook_ muestra el proceso paso a paso y los resultados intermedios de cada técnica aplicada. Al final, se puede realizar una **comparativa visual** entre la imagen original y la imagen procesada, destacando la **mejora significativa** en la calidad y la legibilidad del documento.

---

## 🚀 ¿Cómo Ejecutar el Notebook?

Para ejecutar este proyecto, sigue los siguientes pasos:

1.  Abre el _notebook_ principal: `proyecto_integrador.ipynb`.
2.  Puedes ejecutarlo en **Google Colab** o **Jupyter Notebook**.
3.  Asegúrate de tener las librerías necesarias instaladas. Si usas un entorno local, puedes instalarlas con el siguiente comando:

```bash
pip install opencv-python



