# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from deepface import DeepFace

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Détection d'Émotions Faciales et d'Objets en Temps Réel",
    page_icon="😊",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .title {
        color: #1e81b0;
        font-size: 3em;
        text-align: center;
    }
    .subtitle {
        color: #4682b4;
        font-size: 1.5em;
        text-align: center;
        margin-bottom: 30px;
    }
    .sidebar .sidebar-content {
        background-color: #e6f3ff;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="title">Détection d\'Émotions et d\'Objets</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Application de détection en temps réel</p>', unsafe_allow_html=True)

# Initialisation des modèles
@st.cache_resource
def load_yolo_model():
    return YOLO("./models_detection/yolov8l.pt")

@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Chargement des modèles
yolo_model = load_yolo_model()
face_cascade = load_face_cascade()

# Configuration dans la barre latérale
with st.sidebar:
    st.header("Configuration")
    
    detection_mode = st.radio(
        "Mode de détection",
        ["Émotions faciales", "Objets", "Émotions et Objets"]
    )
    
    if "Objets" in detection_mode:
        confidence = st.slider("Seuil de confiance pour YOLO", 0.0, 1.0, 0.5)
    
    st.markdown("---")
    
    st.info("Cette application combine la détection d'émotions faciales avec DeepFace et la détection d'objets avec YOLOv8.")

# Zone d'affichage principale
col1, col2 = st.columns([3, 1])

with col1:
    run = st.checkbox("Activer la Webcam")
    FRAME_WINDOW = st.image([])

with col2:
    if "Émotions" in detection_mode:
        st.subheader("Émotions détectées")
        emotion_placeholder = st.empty()
    
    if "Objets" in detection_mode:
        st.subheader("Objets détectés")
        objects_placeholder = st.empty()

# Fonction pour la détection des émotions
def detect_emotions(frame):
    # Convertir le cadre en niveaux de gris
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Convertir en RGB pour DeepFace
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
    
    # Détecter les visages
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    emotions_detected = {}
    
    for i, (x, y, w, h) in enumerate(faces):
        try:
            # Extraire le visage
            face = rgb_frame[y:y+h, x:x+w]
            
            # Analyse des émotions
            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            
            # Récupérer l'émotion dominante
            emotion = result[0]['dominant_emotion']
            emotions_detected[f"Visage {i+1}"] = emotion
            
            # Dessiner un rectangle et afficher l'émotion
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as e:
            pass
    
    return frame, emotions_detected

# Fonction pour la détection d'objets
def detect_objects(frame, conf_threshold):
    pil_image = Image.fromarray(frame)
    
    # Détection avec YOLO
    results = yolo_model.predict(pil_image, conf=conf_threshold)
    annotated_frame = results[0].plot(line_width=2, font_size=10)
    
    # Extraction des objets détectés
    detected_objects = {}
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            name = r.names[cls_id]
            if name in detected_objects:
                detected_objects[name] += 1
            else:
                detected_objects[name] = 1
    
    return annotated_frame, detected_objects

# Capture vidéo
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Erreur d'accès à la webcam")
        break
    
    # Conversion BGR à RGB pour affichage dans Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Variables pour stocker les résultats
    emotions_data = {}
    objects_data = {}
    
    # Traitement selon le mode de détection
    if detection_mode == "Émotions faciales":
        processed_frame, emotions_data = detect_emotions(frame.copy())
        FRAME_WINDOW.image(processed_frame)
        
    elif detection_mode == "Objets":
        processed_frame, objects_data = detect_objects(frame.copy(), confidence)
        FRAME_WINDOW.image(processed_frame)
        
    elif detection_mode == "Émotions et Objets":
        # Détection des émotions d'abord
        emotion_frame, emotions_data = detect_emotions(frame.copy())
        
        # Puis détection des objets
        object_frame, objects_data = detect_objects(emotion_frame, confidence)
        
        # Affichage du cadre final
        FRAME_WINDOW.image(object_frame)
    
    # Mise à jour des informations détectées
    if "Émotions" in detection_mode:
        if emotions_data:
            emotion_text = "\n".join([f"{face}: {emotion}" for face, emotion in emotions_data.items()])
            emotion_placeholder.text(emotion_text)
        else:
            emotion_placeholder.text("Aucun visage détecté")
    
    if "Objets" in detection_mode:
        if objects_data:
            objects_text = "\n".join([f"{obj}: {count}" for obj, count in objects_data.items()])
            objects_placeholder.text(objects_text)
        else:
            objects_placeholder.text("Aucun objet détecté")

# Libération des ressources
if not run:
    if cap.isOpened():
        cap.release()
    st.warning("Webcam désactivée")
