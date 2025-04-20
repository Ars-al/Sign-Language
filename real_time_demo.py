# import cv2
# import cv2
# import mediapipe as mp
# import numpy as np
# import joblib  # For loading the trained SVM model

# # Load the trained SVM model
# svm_model = joblib.load('svm_winner.pkl')

# # Initialize Mediapipe Hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # Start video capture
# cap = cv2.VideoCapture(0)

# # Get frame width and height
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))

# # Define codec and create VideoWriter object to save the video
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
# out = cv2.VideoWriter("output.mp4", fourcc, 30, (frame_width, frame_height))

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Flip the frame horizontally for a mirrored effect
#     frame = cv2.flip(frame, 1)

#     # Convert BGR to RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(rgb_frame)

#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             # Extract (x, y, z) coordinates
#             landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

#             # Normalize: Recenter based on wrist position (landmark 0)
#             wrist_x, wrist_y, wrist_z = landmarks[0]
#             landmarks[:, 0] -= wrist_x  
#             landmarks[:, 1] -= wrist_y  

#             # Scale only x and y using the mid-finger tip (landmark 12)
#             mid_finger_x, mid_finger_y, _ = landmarks[12] 
#             scale_factor = np.sqrt(mid_finger_x**2 + mid_finger_y**2)
#             landmarks[:, 0] /= scale_factor  
#             landmarks[:, 1] /= scale_factor  

#             # Flatten the features for SVM
#             features = landmarks.flatten().reshape(1, -1)

#             # Predict using SVM
#             prediction = svm_model.predict(features)[0]

#             # Draw landmarks on the frame
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             # Display the prediction on the frame
#             cv2.putText(frame, f'Prediction: {prediction}', (50, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

#             # Add a bounding box around the hand
#             x_min, x_max = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
#             y_min, y_max = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
#             cv2.rectangle(frame, (int((x_min + 0.1) * frame_width), int((y_min + 0.1) * frame_height)),
#                           (int((x_max - 0.1) * frame_width), int((y_max - 0.1) * frame_height)), (0, 255, 0), 2)

#     # Write the flipped frame to the video file
#     out.write(frame)

#     # Show the flipped frame
#     cv2.imshow("Hand Gesture Recognition", frame)

#     # Exit on pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# out.release()  # Ensure video is saved properly
# cv2.destroyAllWindows()import mediapipe as mp
# import numpy as np
# import joblib  # For loading the trained SVM model

# # Load the trained SVM model
# svm_model = joblib.load('svm_winner.pkl')

# # Initialize Mediapipe Hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # Start video capture
# cap = cv2.VideoCapture(0)

# # Get frame width and height
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))

# # Define codec and create VideoWriter object to save the video
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
# out = cv2.VideoWriter("output.mp4", fourcc, 30, (frame_width, frame_height))

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Flip the frame horizontally for a mirrored effect
#     frame = cv2.flip(frame, 1)

#     # Convert BGR to RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(rgb_frame)

#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             # Extract (x, y, z) coordinates
#             landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

#             # Normalize: Recenter based on wrist position (landmark 0)
#             wrist_x, wrist_y, wrist_z = landmarks[0]
#             landmarks[:, 0] -= wrist_x  
#             landmarks[:, 1] -= wrist_y  

#             # Scale only x and y using the mid-finger tip (landmark 12)
#             mid_finger_x, mid_finger_y, _ = landmarks[12] 
#             scale_factor = np.sqrt(mid_finger_x**2 + mid_finger_y**2)
#             landmarks[:, 0] /= scale_factor  
#             landmarks[:, 1] /= scale_factor  

#             # Flatten the features for SVM
#             features = landmarks.flatten().reshape(1, -1)

#             # Predict using SVM
#             prediction = svm_model.predict(features)[0]

#             # Draw landmarks on the frame
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             # Display the prediction on the frame
#             cv2.putText(frame, f'Prediction: {prediction}', (50, 50),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

#     # Write the flipped frame to the video file
#     out.write(frame)

#     # Show the flipped frame
#     cv2.imshow("Hand Gesture Recognition", frame)

#     # Exit on pressing 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# out.release()  # Ensure video is saved properly
# cv2.destroyAllWindows()


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