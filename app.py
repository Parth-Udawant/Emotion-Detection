import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from utils.emotion_labels import emotion_dict

model = load_model("model/emotion_model.h5")

st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😊",
    layout="centered"
)

st.title("🎭 Real-Time Emotion Detection")
st.markdown(
    "Using your **webcam**, we detect facial **emotions** in real time."
)

st.sidebar.markdown("## Settings")
camera_on = st.sidebar.checkbox("Turn on Camera", value=False)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='text-align: center;'>Made with ❤️ by <a href='https://instagram.com/theidealcoder'>@theidealcoder</a></p>",
    unsafe_allow_html=True
)

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    predictions = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi_gray, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        prediction = model.predict(roi)[0]
        label = emotion_dict[prediction.argmax()]
        predictions.append((x, y, w, h, label))

    return predictions

if camera_on:
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)
    st.success("Webcam is ON. Showing real-time emotion predictions...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Unable to access camera.")
            break

        frame = cv2.flip(frame, 1)
        predictions = detect_emotion(frame)

        for (x, y, w, h, label) in predictions:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    cap.release()
else:
    st.info("☝️ Turn on the camera from the sidebar to begin.")
