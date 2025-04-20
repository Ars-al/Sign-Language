# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# import joblib

# # Load your dataset
# df = pd.read_csv('custom_hand_signs.csv')

# # Separate features and labels
# X = df.drop('label', axis=1).values
# y = df['label'].values

# # Split into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Train an SVM classifier
# svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
# svm_model.fit(X_train, y_train)

# # Evaluate the model
# train_pred = svm_model.predict(X_train)
# test_pred = svm_model.predict(X_test)

# print(f"Training Accuracy: {accuracy_score(y_train, train_pred):.2f}")
# print(f"Test Accuracy: {accuracy_score(y_test, test_pred):.2f}")

# # Save the model and scaler
# joblib.dump(svm_model, 'custom_gesture_model.pkl')
# joblib.dump(scaler, 'custom_gesture_scaler.pkl')


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

def load_and_preprocess_data(filename):
    """Load and preprocess the gesture data"""
    df = pd.read_csv(filename)
    
    # Check for missing data
    if df.isnull().any().any():
        print("Warning: Missing data found. Dropping rows with missing values.")
        df = df.dropna()
    
    # Separate features and labels
    X = df.iloc[:, :-1]  # All columns except last
    y = df.iloc[:, -1]   # Last column (labels)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X.values, y_encoded, le

def extract_features(landmarks):
    """Extract enhanced features from raw landmarks"""
    landmarks = landmarks.reshape(-1, 3)  # Reshape to 21x3
    
    # 1. Relative to wrist (landmark 0)
    wrist = landmarks[0]
    relative = landmarks - wrist
    
    # 2. Distances between key points
    fingertips = [4, 8, 12, 16, 20]  # Thumb to pinky
    distances = []
    for i in fingertips:
        distances.append(np.linalg.norm(landmarks[i] - wrist))
    
    # 3. Angles between fingers
    angles = []
    vectors = []
    for i in fingertips:
        vectors.append(landmarks[i] - wrist)
    
    for i in range(len(vectors)-1):
        cos_angle = np.dot(vectors[i], vectors[i+1]) / (
            np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[i+1]))
        angles.append(np.arccos(np.clip(cos_angle, -1, 1)))
    
    # Combine all features
    features = np.concatenate([
        relative.flatten(),  # Original relative coordinates
        np.array(distances),  # Distances
        np.array(angles)     # Angles
    ])
    
    return features

def train_model(X, y, test_size=0.2, random_state=42):
    """Train and evaluate the gesture recognition model"""
    # Enhanced feature extraction
    X_enhanced = np.array([extract_features(x) for x in X])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y, test_size=test_size, random_state=random_state)
    
    # Create preprocessing pipeline
    preprocessor = make_pipeline(
        StandardScaler()
    )
    
    # Try different models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, 
                                             max_depth=10,
                                             class_weight='balanced',
                                             random_state=42),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale', 
                  probability=True, random_state=42)
    }
    
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        # Create full pipeline
        pipeline = make_pipeline(
            preprocessor,
            model
        )
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)
        
        print(f"\n{name} Performance:")
        print(f"Training Accuracy: {train_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        
        # Detailed report
        y_pred = pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))
        
        # Track best model
        if test_score > best_score:
            best_score = test_score
            best_model = pipeline
    
    return best_model

def main():
    # Load data
    X, y, label_encoder = load_and_preprocess_data('custom_hand_signs.csv')
    
    # Train model
    model = train_model(X, y)
    
    # Save the trained model and label encoder
    joblib.dump(model, 'gesture_recognition_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("\nModel training complete and saved!")

if __name__ == "__main__":
    main()