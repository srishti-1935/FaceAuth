# Face Authentication System

Real-time face recognition with emotion detection and access control.

## Problem Statement
Traditional passwords can be stolen or forgotten. This project provides secure authentication using face recognition and emotion analysis.

## Tech Stack
- OpenCV for face detection
- DeepFace for emotion analysis  
- Streamlit for web interface
- Python for backend logic

## Steps to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Add face images in data/person1, data/person2, data/person3 folders
3. Train model: `python train_model.py`
4. Run app: `streamlit run app.py`

## Features
- Real-time face recognition
- Emotion detection (happy, sad, angry, etc.)
- Access granted/denied with confidence scores
- Multi-user support

## Project Structure
- app.py - Main application
- train_model.py - Model training
- utils/face_detector.py - Face recognition logic
- data/ - Training images
- models/ - Saved models

## Author
[Srishti Srivastava]