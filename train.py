# train.py
import os, cv2, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
import joblib

TRAIN_PATH = "train"
IMG_SIZE = 64  # Smaller size forces model to look at strokes, not layout
EPOCHS = 26    # 23 was the optimal epochs for the smaller model
BATCH = 8      # Smaller batch for better convergence 

X, y = [], []

print("📦 Extracting Handwriting Textures...")
for fname in sorted(os.listdir(TRAIN_PATH)):
    if not fname.lower().endswith(".png"): continue
    label = fname[:2]
    img = cv2.imread(os.path.join(TRAIN_PATH, fname), cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # We slice the image into many small 64x64 patches
    h, w = img.shape
    for i in range(0, h-IMG_SIZE, 32): # Overlapping slices
        for j in range(0, w-IMG_SIZE, 32):
            patch = img[i:i+IMG_SIZE, j:j+IMG_SIZE]
            patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
            X.append(patch / 255.0)
            y.append(label)

X = np.array(X)
le = LabelEncoder()
y = le.fit_transform(y)
joblib.dump(le, "labels.pkl")

print(f"🔥 Training on {len(X)} handwriting texture samples...")

# A custom "Forensic" CNN
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=EPOCHS, batch_size=BATCH, verbose=1)
model.save("writer_model.keras")
print("✅ Forensic Model Saved")