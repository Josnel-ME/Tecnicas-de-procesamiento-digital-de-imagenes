"""
Analizador de expresiones faciales mejorado y estable.
Basado en landmarks de MediaPipe Face Mesh.
"""

import math
import numpy as np
from collections import deque


class ExpressionAnalyzer:
    """
    Analiza expresiones faciales y devuelve métricas normalizadas
    + interpretación de expresión facial básica.
    """

    def __init__(self, smooth_window=5):
        self.history = {
            'mouth_opening': deque(maxlen=smooth_window),
            'eye_opening': deque(maxlen=smooth_window),
            'head_tilt': deque(maxlen=smooth_window),
        }

    # --------------------------------------------------------
    # MÉTODO PRINCIPAL
    # --------------------------------------------------------
    def analyze(self, landmarks, image_shape):
        alto, ancho = image_shape

        # Calcular métricas base
        raw = {
            'mouth_opening': self._calculate_mouth_opening(landmarks, alto, ancho),
            'eye_opening': self._calculate_eye_opening(landmarks, alto, ancho),
            'head_tilt': self._calculate_head_tilt(landmarks, alto, ancho)
        }

        # Suavizado temporal
        for k, v in raw.items():
            self.history[k].append(v)
        smoothed = {k: float(np.mean(self.history[k])) for k in self.history}

        # Interpretación
        interpretation = self._interpret(smoothed)

        return {
            "metrics": smoothed,
            "interpretation": interpretation
        }

    # --------------------------------------------------------
    # CÁLCULOS DE MÉTRICAS
    # --------------------------------------------------------
    def _get_eye_distance(self, landmarks, alto, ancho):
        """Distancia entre ojos, usada para normalizar métricas."""
        try:
            left_eye = landmarks.landmark[133]
            right_eye = landmarks.landmark[362]
            dx = (right_eye.x - left_eye.x) * ancho
            dy = (right_eye.y - left_eye.y) * alto
            return math.sqrt(dx**2 + dy**2) or 1e-6
        except Exception:
            return 1.0

    def _calculate_mouth_opening(self, landmarks, alto, ancho):
        """Apertura de la boca normalizada por distancia entre ojos."""
        try:
            upper = landmarks.landmark[13]
            lower = landmarks.landmark[14]
            y_upper = upper.y * alto
            y_lower = lower.y * alto
            mouth_opening_px = abs(y_lower - y_upper)
            eye_dist = self._get_eye_distance(landmarks, alto, ancho)
            return mouth_opening_px / eye_dist
        except Exception:
            return 0.0

    def _calculate_eye_opening(self, landmarks, alto, ancho):
        """Apertura promedio de ambos ojos, normalizada."""
        try:
            # Izquierdo
            up_l = landmarks.landmark[159]
            low_l = landmarks.landmark[145]
            # Derecho
            up_r = landmarks.landmark[386]
            low_r = landmarks.landmark[374]

            left_open = abs(low_l.y - up_l.y) * alto
            right_open = abs(low_r.y - up_r.y) * alto
            avg_open = (left_open + right_open) / 2.0
            eye_dist = self._get_eye_distance(landmarks, alto, ancho)
            return avg_open / eye_dist
        except Exception:
            return 0.0

    def _calculate_head_tilt(self, landmarks, alto, ancho):
        """Inclinación de cabeza (°) basada en línea entre ojos."""
        try:
            left = landmarks.landmark[133]
            right = landmarks.landmark[362]
            dx = (right.x - left.x) * ancho
            dy = (right.y - left.y) * alto
            angle_rad = math.atan2(dy, dx)
            return round(math.degrees(angle_rad), 2)
        except Exception:
            return 0.0

    # --------------------------------------------------------
    # INTERPRETACIÓN DE MÉTRICAS
    # --------------------------------------------------------
    def _interpret(self, m):
        """
        Devuelve una interpretación textual de las métricas.
        Los umbrales pueden ajustarse según tus datos.
        """
        mouth = m["mouth_opening"]
        eyes = m["eye_opening"]
        tilt = m["head_tilt"]

        # --- Interpretación de boca ---
        if mouth < 0.14:
            mouth_state = "Boca cerrada – expresión neutral o seria"
        elif mouth < 0.24:
            mouth_state = "Boca semiabierta – ligera sonrisa o habla"
        else:
            mouth_state = "Boca abierta – sorpresa o risa"

        # --- Interpretación de ojos ---
        if eyes < 0.10:
            eyes_state = "Ojos cerrados – posible parpadeo o sueño"
        elif eyes < 0.17:
            eyes_state = "Ojos entrecerrados – concentración o luz intensa"
        else:
            eyes_state = "Ojos abiertos – atención o sorpresa"

        # --- Interpretación de inclinación ---
        if abs(tilt) < 5:
            head_state = "Cabeza recta – neutral"
        elif tilt > 5:
            head_state = "Cabeza inclinada a la derecha"
        else:
            head_state = "Cabeza inclinada a la izquierda"

        return {
            "mouth": mouth_state,
            "eyes": eyes_state,
            "head": head_state
        }
