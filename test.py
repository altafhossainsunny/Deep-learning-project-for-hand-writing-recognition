import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
from scipy.spatial.distance import euclidean

# --- Configuration ---
train_data_path = r"C:\Users\MD ALTAF HOSSAIN\PycharmProjects\Deeplearning project\Deep-learning-project-for-hand-writing-recognition\train"
test_data_path = r"C:\Users\MD ALTAF HOSSAIN\PycharmProjects\Deeplearning project\Deep-learning-project-for-hand-writing-recognition\test"
IMG_SIZE = 128

# 1. Load the trained Siamese Encoder
print("Loading Encoder...")
model = load_model('siamese_encoder.h5', compile=False)

def get_single_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        thresh = thresh[y:y+h, x:x+w]
    img_resized = cv2.resize(thresh, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    return img_rgb / 255.0

# 2. Create the "Reference Gallery"
print("Building Gallery (extracting features for 70 writers)...")
gallery_vectors = []
gallery_labels = []

train_files = sorted([f for f in os.listdir(train_data_path) if f.endswith('.png')])
for filename in train_files:
    img = get_single_image(os.path.join(train_data_path, filename))
    if img is not None:
        # Extract the "fingerprint" of the handwriting
        vector = model.predict(np.expand_dims(img, 0), verbose=0)
        gallery_vectors.append(vector.flatten())
        gallery_labels.append(int(filename[0:2]))

# 3. Test and Compare
print(f"Testing {len(os.listdir(test_data_path))} images...")
y_true = []
y_pred = []

test_files = sorted([f for f in os.listdir(test_data_path) if f.endswith('.png')])

for filename in test_files:
    test_img = get_single_image(os.path.join(test_data_path, filename))
    if test_img is not None:
        actual_writer = int(filename[0:2])
        
        # Extract features for test image
        test_vector = model.predict(np.expand_dims(test_img, 0), verbose=0).flatten()
        
        # Find the smallest distance in the gallery
        distances = [euclidean(test_vector, ref_vec) for ref_vec in gallery_vectors]
        best_match_idx = np.argmin(distances)
        predicted_writer = gallery_labels[best_match_idx]
        
        y_true.append(actual_writer)
        y_pred.append(predicted_writer)

# 4. Final Performance Report
print("\n" + "="*40)
print("SIAMESE ONE-SHOT RESULTS")
print("="*40)
print(f"Final Accuracy: {accuracy_score(y_true, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, zero_division=0))