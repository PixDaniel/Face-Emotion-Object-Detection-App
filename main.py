# main.py
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Face Emotion Recognition & Object Detection In Real-Time",
    page_icon="😊",
    layout="wide"
)

st.markdown("""
    <style>
    .title {
        color: #1e81b0;
        font-size: 3em;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background-color: #e6f3ff;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="title">Face Emotion Recognition</h1>', unsafe_allow_html=True)

# Initialisation du modèle
@st.cache_resource
def load_model():
    return YOLO("/models_detection/yolov8l.pt")

model = load_model()

with st.sidebar:
    st.header("Configuration")
    confidence = st.slider("Seuil de confiance", 0.0, 1.0, 0.5)
    st.markdown("---")
    st.info("Cette application détecte les émotions en temps réel à l'aide d'un modèle YOLO entraîné.")

# Gestion de la webcam
run = st.checkbox("Activer la webcam")
FRAME_WINDOW = st.image([])

# Capture vidéo
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Erreur d'accès à la webcam")
        break
    
    # Conversion des couleurs
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame)
    
    # Détection avec YOLO
    results = model.predict(pil_image, conf=confidence)
    annotated_frame = results[0].plot(line_width=2, font_size=10)
    
    # Affichage du résultat
    FRAME_WINDOW.image(annotated_frame)

if not run:
    cap.release()
    st.warning("Webcam désactivée")