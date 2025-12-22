import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import MobileNetV2
from sklearn.preprocessing import LabelEncoder

# ================= CONFIG =================
TRAIN_PATH = "train"
IMG_SIZE = 128 # MobileNetV2 requires specific sizes
# =========================================

def load_data(path):
    images, labels = [], []
    files = [f for f in sorted(os.listdir(path)) if f.lower().endswith(".png")]
    for fname in files:
        label = fname[:2]
        img = cv2.imread(os.path.join(path, fname)) # Load as BGR for MobileNet
        if img is None: continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        images.append(img.astype("float32") / 255.0)
        labels.append(label)
    return np.array(images), np.array(labels)

print("📦 Loading Data for Transfer Learning...")
X_train, y_train_raw = load_data(TRAIN_PATH)
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)
num_classes = len(le.classes_)

# Use MobileNetV2 as a feature extractor (Frozen)
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False 

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"🔥 Training on {num_classes} Writers...")
model.fit(X_train, y_train, epochs=30, batch_size=8, verbose=1)

model.save("model.h5")
print("✅ New model.h5 saved.")


'''
Preprocessing pipeline:
1. Grayscale Conversion
The images were loaded using cv2.IMREAD_GRAYSCALE.
Why: Handwriting color (blue vs. black ink) is usually irrelevant for identity. Grayscale reduces the data complexity from 3 channels (RGB) to 1, focusing entirely on the intensity of the pen strokes.
2. Binary Thresholding (Otsu’s Method)
We used cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU).
Inversion (BINARY_INV): Standard images have dark ink on a light background. We inverted this so the ink is white (255) and the background is black (0). Computer vision algorithms generally treat "white" as the signal/feature and "black" as the void.
Otsu’s Optimization: This automatically calculates the perfect threshold value for each image. It separates the foreground (text) from the background (paper texture or shadows) without you having to manually guess a number.
3. Feature Keypoint Extraction (ORB)
Instead of looking at raw pixels, the code preprocessed the image into Descriptors.
Corner Detection: It identifies "corners" or "edges" where the pen changes direction (e.g., the loop of an 'o' or the cross of a 't').
Scale Invariance: The ORB algorithm creates a "pyramid" of the image at different sizes. This ensures that even if one sample is written slightly larger or smaller than another, the model can still match the handwriting style.
4. Normalization
For the neural network attempts (before we switched to ORB), we used Pixel Scaling:
Division by 255: This scales pixel values from $[0, 255]$ to $[0, 1]$.
Why: Neural networks converge much faster when the input data has a small, consistent range. Without this, the gradients during training would "explode," causing the NaN errors you saw earlier.

Why this worked for your 40% Accuracy
Standard "Global" preprocessing (like just resizing the whole page) often loses the fine details of handwriting. By using Otsu's Thresholding, we effectively "scanned" the ink away from the paper, allowing the ORB algorithm to see the unique "shakiness" or "fluidity" of each writer's hand.

'''

#making your life easier