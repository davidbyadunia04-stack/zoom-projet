import streamlit as st
import cv2
import numpy as np

st.title("Mon Suivi Caméra CapCut")

# Détecteur de visage
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Caméra Web
img_file_buffer = st.camera_input("Autorise la caméra")

if img_file_buffer:
    # Lecture de l'image
    bytes_data = img_file_buffer.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Détection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        # On crée un cadre large (effet CapCut)
        zoom_w, zoom_h = w * 3, h * 3
        cx, cy = x + w // 2, y + h // 2

        x1 = max(0, cx - zoom_w // 2)
        y1 = max(0, cy - zoom_h // 2)

        crop = frame[y1:y1 + zoom_h, x1:x1 + zoom_w]
        if crop.size > 0:
            st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)
    else:
        st.write("Recherche de visage...")