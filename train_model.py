import os
import cv2
import numpy as np
import pickle

print("Starting simple face training...")

known_faces = []
known_names = []

for person_name in os.listdir("data"):
    person_folder = os.path.join("data", person_name)
    
    if os.path.isdir(person_folder):
        print(f"Loading {person_name}'s photos...")
        
        for photo_name in os.listdir(person_folder):
            if photo_name.endswith(('.jpg', '.png')):
                photo_path = os.path.join(person_folder, photo_name)
                
                known_faces.append(photo_path)
                known_names.append(person_name)
                print(f"    Loaded {photo_name}")


model_data = {'face_paths': known_faces, 'names': known_names}

with open('models/simple_face_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f" Training complete! Loaded {len(known_faces)} face images.")