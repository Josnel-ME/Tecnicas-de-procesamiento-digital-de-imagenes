# app.py
"""
Aplicación Streamlit para detección de landmarks faciales.
"""
import streamlit as st
from PIL import Image
from src.detector import FaceLandmarkDetector
from src.utils import pil_to_cv2, cv2_to_pil, resize_image
from src.config import TOTAL_LANDMARKS


# Configuración de la página
st.set_page_config(
    page_title="Detector de Landmarks Faciales",
    layout="wide"
)

# Menú principal
menu = st.sidebar.selectbox(
    "Selecciona una opción",
    ["Detector de Landmarks", "Análisis de Expresiones"]
)

if menu == "Detector de Landmarks":
    # Título y descripción
    st.title("Detector de Landmarks Faciales")
    st.markdown("""
    Esta aplicación detecta **478 puntos clave** en rostros humanos usando MediaPipe.
    Subí una imagen con un rostro y mirá la magia de la visión por computadora.
    """)
elif menu == "Análisis de Expresiones":
    st.title("Análisis de Expresiones Faciales")
    st.markdown("""
    Analiza expresiones faciales calculando métricas como apertura de boca, ojos e inclinación de cabeza.
    """)

# Sidebar con información
with st.sidebar:
    st.header("Información")
    st.markdown("""
    ### ¿Qué son los Landmarks?
    Son puntos de referencia que mapean:
    - Ojos (iris, párpados)
    - Nariz (puente, fosas)
    - Boca (labios, comisuras)
    - Contorno facial
    
    ### Aplicaciones
    - Filtros AR (Instagram)
    - Análisis de expresiones
    - Animación facial
    - Autenticación biométrica
    """)
    
    st.divider()
    st.caption("Desarrollado en el Laboratorio 2 - IFTS24")

if menu == "Detector de Landmarks":
    # Opciones de visualización
    visualization_mode = st.selectbox(
        "Modo de visualización",
        ["Puntos", "Puntos + Malla", "Contornos principales", "Heatmap"],
        help="Elegí cómo visualizar los landmarks faciales"
    )

    # Uploader de imagen
    uploaded_file = st.file_uploader(
        "Subí una imagen con un rostro",
        type=["jpg", "jpeg", "png"],
        help="Formatos aceptados: JPG, JPEG, PNG"
    )
elif menu == "Análisis de Expresiones":
    # Uploader de imagen para análisis de expresiones
    uploaded_file = st.file_uploader(
        "Subí una imagen con un rostro para análisis",
        type=["jpg", "jpeg", "png"],
        help="Formatos aceptados: JPG, JPEG, PNG"
    )

if uploaded_file is not None:
    # Cargar imagen
    imagen_original = Image.open(uploaded_file)

    # Convertir a formato OpenCV
    imagen_cv2 = pil_to_cv2(imagen_original)

    # Redimensionar si es muy grande
    imagen_cv2 = resize_image(imagen_cv2, max_width=800)

    if menu == "Detector de Landmarks":
        # Columnas para mostrar antes/después
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Imagen Original")
            st.image(cv2_to_pil(imagen_cv2))

        # Detectar landmarks
        with st.spinner("Detectando landmarks faciales..."):
            detector = FaceLandmarkDetector()
            imagen_procesada, landmarks, info = detector.detect(imagen_cv2, visualization_mode)
            detector.close()

        with col2:
            st.subheader("Landmarks Detectados")
            st.image(cv2_to_pil(imagen_procesada))

        # Mostrar información de detección
        st.divider()

        if info["deteccion_exitosa"]:
            st.success("Detección exitosa")

            # Métricas
            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric("Rostros detectados", info["rostros_detectados"])

            with metric_col2:
                st.metric("Landmarks detectados", f"{info['total_landmarks']}/{TOTAL_LANDMARKS}")

            with metric_col3:
                porcentaje = (info['total_landmarks'] / TOTAL_LANDMARKS) * 100
                st.metric("Precisión", f"{porcentaje:.1f}%")
        else:
            st.error("No se detectó ningún rostro en la imagen")
            st.info("""
            **Consejos**:
            - Asegurate de que el rostro esté bien iluminado
            - El rostro debe estar mirando hacia la cámara
            - Probá con una imagen de mayor calidad
            """)

    elif menu == "Análisis de Expresiones":
        # Mostrar imagen original
        st.subheader("Imagen Analizada")
        st.image(cv2_to_pil(imagen_cv2))

        # Detectar landmarks para análisis
        with st.spinner("Analizando expresión facial..."):
            detector = FaceLandmarkDetector()
            _, landmarks, info = detector.detect(imagen_cv2, "Puntos")  # Solo necesitamos landmarks
            detector.close()

        if info["deteccion_exitosa"] and landmarks:
            st.success("Análisis completado")

            # Calcular métricas de expresión
            from src.expressions import ExpressionAnalyzer
            analyzer = ExpressionAnalyzer()
            result = analyzer.analyze(landmarks, imagen_cv2.shape[:2])
            metrics = result["metrics"]
            interpret = result["interpretation"]


            # Mostrar métricas
            st.divider()
            st.subheader("Métricas de Expresión Facial")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Apertura de Boca", f"{metrics['mouth_opening']:.2f}px")

            with col2:
                st.metric("Apertura de Ojos", f"{metrics['eye_opening']:.2f}px")

            with col3:
                st.metric("Inclinación de Cabeza", f"{metrics['head_tilt']:.1f}°")

            # Interpretación
            st.subheader("Interpretación")
            st.info(interpret["mouth"])
            st.info(interpret["eyes"])
            st.info(interpret["head"])
        else:
            st.error("No se pudo analizar la expresión facial")
            st.info("Asegurate de que haya un rostro claramente visible en la imagen")

else:
    if menu == "Detector de Landmarks":
        # Mensaje de bienvenida
        st.info("Subí una imagen para comenzar la detección")

        # Ejemplo visual
        st.markdown("### Ejemplo de Resultado")
        st.image(
            "https://ai.google.dev/static/mediapipe/images/solutions/face_landmarker_keypoints.png?hl=es-419",
            caption="MediaPipe detecta 478 landmarks faciales",
            width=400
        )
    elif menu == "Análisis de Expresiones":
        st.info("Subí una imagen para analizar la expresión facial")
        st.markdown("### Métricas Disponibles")
        st.markdown("- **Apertura de boca**: Distancia vertical entre labios")
        st.markdown("- **Apertura de ojos**: Promedio de apertura de ambos ojos")
        st.markdown("- **Inclinación de cabeza**: Ángulo de rotación de la cabeza")