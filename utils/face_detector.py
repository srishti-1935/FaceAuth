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
        except Exception as e:
            print(f"Model loading error: {e}")
            self.known_faces = []
            self.known_names = []
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def recognize_face(self, image):
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = self.face_cascade.detectMultiScale(rgb_image, 1.1, 4)
            
            results = []
            
            for (x, y, w, h) in face_locations:
                face_roi = image[y:y+h, x:x+w]
                
                identity, confidence = self.simple_face_match(face_roi)
                
                if confidence > 0.6:
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
        except Exception as e:
            print(f"Face recognition error: {e}")
            return []
    
    def simple_face_match(self, face_image):
        best_match = "Unknown"
        best_confidence = 0
        
        if not self.known_faces:
            return best_match, best_confidence
        
        current_face = cv2.resize(face_image, (100, 100))
        
        for i, known_face_path in enumerate(self.known_faces):
            try:
                known_face = cv2.imread(known_face_path)
                if known_face is None:
                    continue
                    
                known_face = cv2.resize(known_face, (100, 100))
                
                hist1 = cv2.calcHist([current_face], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([known_face], [0], None, [256], [0, 256])
                
                similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                confidence = max(0, (similarity + 1) / 2)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = self.known_names[i]
            except Exception:
                continue
        
        return best_match, round(best_confidence, 2)
    
    def detect_emotion(self, face_image):
        try:
            temp_path = "temp_face.jpg"
            cv2.imwrite(temp_path, face_image)
            
            analysis = DeepFace.analyze(temp_path, actions=['emotion'], enforce_detection=False)
            
            if analysis and len(analysis) > 0:
                emotion = analysis[0]['dominant_emotion']
                emotion_score = analysis[0]['emotion'][emotion] / 100
                return emotion, round(emotion_score, 2)
                
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            print(f"Emotion detection error: {e}")
        
        return "neutral", 0.5