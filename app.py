import streamlit as st
import cv2
import numpy as np
from utils.face_detector import SimpleFaceDetector
import time

st.set_page_config(page_title="FaceLock", layout="wide")
st.title("FaceLock - Face Authentication")
st.markdown("### Your face is your password")

def load_detector():
    return SimpleFaceDetector()

detector = load_detector()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Camera")
    start_camera = st.button("Start Camera")
    stop_camera = st.button("Stop Camera")
    
    if start_camera:
        camera_placeholder = st.empty()
        status_placeholder = st.empty()
        
        cap = cv2.VideoCapture(0)
        stop_flag = False
        
        while not stop_flag:
            ret, frame = cap.read()
            if not ret:
                st.error("Cannot access camera")
                break
            
            frame = cv2.flip(frame, 1)
            results = detector.recognize_face(frame)
            
            for result in results:
                top, right, bottom, left = result['location']
                cv2.rectangle(frame, (left, top), (right, bottom), result['color'], 2)
                label = f"{result['name']} ({result['confidence']})"
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), result['color'], cv2.FILLED)
                cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB")
            
            if results:
                result = results[0]
                if result['status'] == "Access Granted":
                    status_placeholder.success(f"{result['status']} - Welcome {result['name']}!")
                else:
                    status_placeholder.error(f"{result['status']} - Face not recognized")
                
                st.info(f"Emotion: {result['emotion']} ({result['emotion_score']})")
            
            time.sleep(0.1)
            
            if stop_camera:
                stop_flag = True
                cap.release()

with col2:
    st.subheader("System Info")
    st.success("Status: Ready")
    st.info(f"Known People: {len(detector.known_names)}")
    
    st.subheader("How to Use")
    st.markdown("1. Click Start Camera")
    st.markdown("2. Look at the camera")
    st.markdown("3. Green box = Access Granted")
    st.markdown("4. Red box = Access Denied")