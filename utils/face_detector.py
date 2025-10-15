import cv2
import pickle
import numpy as np
from deepface import DeepFace
import os

class SimpleFaceDetector:
    def __init__(self):
        try:
            with open('models/simple_face_model.pkl', 'rb') as f:
                data = pickle.load(f)
                self.known_faces = data['face_paths']
                self.known_names = data['names']
        except:
            self.known_faces = []
            self.known_names = []
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def recognize_face(self, image):
        """Simple face recognition using OpenCV and image comparison"""
       
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        results = []
        
        for (x, y, w, h) in faces:
            face_roi = image[y:y+h, x:x+w]
            
            identity, confidence = self.simple_face_match(face_roi)
            
            if confidence > 0.5:
                name = identity
                status = "Access Granted"
                color = (0, 255, 0) 
            else:
                name = "Unknown"
                status = "Access Denied"
                color = (0, 0, 255)  
            
            emotion, emotion_score = self.detect_emotion(face_roi)
            
            results.append({
                'name': name,
                'confidence': confidence,
                'status': status,
                'location': (y, x+w, y+h, x),  
                'color': color,
                'emotion': emotion,
                'emotion_score': emotion_score
            })
        
        return results
    
    def simple_face_match(self, face_image):
        """Simple face matching using OpenCV template matching"""
        best_match = "Unknown"
        best_confidence = 0
        
        current_face = cv2.resize(face_image, (100, 100))
        
        for i, known_face_path in enumerate(self.known_faces):
            try:
                known_face = cv2.imread(known_face_path)
                known_face = cv2.resize(known_face, (100, 100))
                
                hist1 = cv2.calcHist([current_face], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([known_face], [0], None, [256], [0, 256])
                
                similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                confidence = max(0, (similarity + 1) / 2) 
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = self.known_names[i]
            except:
                continue
        
        return best_match, round(best_confidence, 2)
    
    def detect_emotion(self, face_image):
        """Detect emotion using DeepFace"""
        try:
            temp_path = "temp_face.jpg"
            cv2.imwrite(temp_path, face_image)
            
            analysis = DeepFace.analyze(temp_path, actions=['emotion'], enforce_detection=False)
            
            if analysis and len(analysis) > 0:
                emotion = analysis[0]['dominant_emotion']
                emotion_score = analysis[0]['emotion'][emotion] / 100
                return emotion, round(emotion_score, 2)
        except Exception as e:
            print(f"Emotion detection error: {e}")
        
        return "Unknown", 0