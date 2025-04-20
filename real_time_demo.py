import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque

# Load trained model and label encoder
model = joblib.load('gesture_recognition_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# For temporal smoothing
prediction_history = deque(maxlen=5)  # Stores last 5 predictions

def extract_features(landmarks):
    """Same feature extraction as during training"""
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # 1. Relative to wrist
    wrist = landmarks[0]
    relative = landmarks - wrist
    
    # 2. Distances between key points
    fingertips = [4, 8, 12, 16, 20]
    distances = []
    for i in fingertips:
        distances.append(np.linalg.norm(landmarks[i] - wrist))
    
    # 3. Angles between fingers
    vectors = []
    for i in fingertips:
        vectors.append(landmarks[i] - wrist)
    
    angles = []
    for i in range(len(vectors)-1):
        cos_angle = np.dot(vectors[i], vectors[i+1]) / (
            np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[i+1]))
        angles.append(np.arccos(np.clip(cos_angle, -1, 1)))
    
    # Combine features
    features = np.concatenate([
        relative.flatten(),
        np.array(distances),
        np.array(angles)
    ])
    
    return features

def main():
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Flip and convert color
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract and process features
                features = extract_features(hand_landmarks.landmark)
                features = features.reshape(1, -1)
                
                # Predict
                proba = model.predict_proba(features)[0]
                pred_index = np.argmax(proba)
                confidence = proba[pred_index]
                
                # Only accept predictions with high confidence
                if confidence > 0.7:
                    prediction = label_encoder.inverse_transform([pred_index])[0]
                    prediction_history.append(prediction)
                    
                    # Get most frequent recent prediction
                    if len(prediction_history) > 0:
                        final_pred = max(set(prediction_history), 
                                       key=prediction_history.count)
                        cv2.putText(image, f"{final_pred} ({confidence:.2f})", 
                                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                   1, (0, 255, 0), 2)
        
        cv2.imshow('Gesture Recognition', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()