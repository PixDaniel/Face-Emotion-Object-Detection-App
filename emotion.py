import cv2
from deepface import DeepFace

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '/models_detection/haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    # Capture image par image
    ret, frame = cap.read()

    # Convertir le cadre en niveaux de gris
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Conversion de la trame en niveaux de gris au format couleur (RGB)
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Détecter les visages dans le cadre
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extraire le visage de le cadre
        face = rgb_frame[y:y + h, x:x + w]
        
        # Effectuer une analyse des émotions sur le visage
        result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)

        # Déterminer l'émotion dominante
        emotion = result[0]['dominant_emotion']

        # Dessinez un rectangle autour du visage et indiquez l'émotion prédite.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Afficher le cadre résultant
    cv2.imshow('Real-time Emotion Detection', frame)

    # Appuyez sur “q” pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Relâchez la capture et fermez toutes les fenêtres
cap.release()
cv2.destroyAllWindows()