  
import cv2
import streamlit as st
from deepface import DeepFace

def analyze_frame(frame, selected_actions):
    if not selected_actions:
        return frame

    results = DeepFace.analyze(img_path=frame, actions=selected_actions, enforce_detection=False)
    for i in results:
        face_region = i['region']
        dominant_emotion = i.get('dominant_emotion', 'N/A')
        dominant_age = i.get('age', 'N/A')
        dominant_gender = i.get('dominant_gender', 'N/A')

        x, y, h, w = face_region['x'], face_region['y'], face_region['w'], face_region['h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        if 'emotion' in selected_actions:
            text_emotion = f"Emotion: {dominant_emotion}"
            cv2.putText(frame, text_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        if 'age' in selected_actions:
            text_age = f"Age: {dominant_age}"
            cv2.putText(frame, text_age, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        
        if 'gender' in selected_actions:
            text_gender = f"Gender: {dominant_gender}"
            cv2.putText(frame, text_gender, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return frame

def main():
    st.title("Face Analysis with DeepFace and OpenCV")

    cap = cv2.VideoCapture(0)
    selected_actions = st.multiselect("Select Face Analysis Actions", ['emotion', 'age', 'gender'], default=['emotion', 'age', 'gender'])
    new = st.button("Analyze")
    frame_placeholder = st.empty()  # Placeholder for the st.image widget
    warning_placeholder =st.empty()

    while True:
        ret, frame = cap.read()

        if new:
            if not selected_actions:
                warning_placeholder.warning("Please select at least one face analysis action.")
            else:
                frame = analyze_frame(frame, selected_actions)
                
        frame_placeholder.image(frame, channels="BGR", caption="Webcam", use_column_width=True)

if __name__ == "__main__":
    main()
