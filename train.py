import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import tensorflow.keras.backend as K

# --- Configuration ---
data_path = r"C:\Users\MD ALTAF HOSSAIN\PycharmProjects\Deeplearning project\Deep-learning-project-for-hand-writing-recognition\train"
IMG_SIZE = 128

# --- 1. Preprocessing & Pair Generation ---
def get_image(path):
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

def create_pairs(folder_path):
    filenames = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    writer_images = [get_image(os.path.join(folder_path, f)) for f in filenames]
    num_writers = len(writer_images)
    
    pair_images_a = []
    pair_images_b = []
    pair_labels = []

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1
    )

    print("Generating Pairs...")
    for i in range(num_writers):
        current_img = writer_images[i]
        # Positive Pairs
        aug_iter = datagen.flow(np.expand_dims(current_img, 0), batch_size=1)
        for _ in range(10): 
            pair_images_a.append(current_img)
            pair_images_b.append(next(aug_iter)[0])
            pair_labels.append(1.0)

        # Negative Pairs
        for _ in range(10):
            j = np.random.randint(0, num_writers)
            while i == j: j = np.random.randint(0, num_writers)
            pair_images_a.append(current_img)
            pair_images_b.append(writer_images[j])
            pair_labels.append(0.0)

    return [np.array(pair_images_a), np.array(pair_images_b)], np.array(pair_labels)

X_pairs, y_pairs = create_pairs(data_path)

# --- 2. Siamese Architecture ---
def create_base_network():
    return models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu')
    ], name="base_network")

base_network = create_base_network()

input_a = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
input_b = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

feat_a = base_network(input_a)
feat_b = base_network(input_b)

# --- FIX: Using tf functions directly for better compatibility ---
def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))

distance = layers.Lambda(euclidean_distance)([feat_a, feat_b])
siamese_model = models.Model(inputs=[input_a, input_b], outputs=distance)

# --- FIX: Robust Contrastive Loss ---
def contrastive_loss(y_true, y_pred):
    margin = 1.0
    y_true = tf.cast(y_true, tf.float32)
    # If same (1), minimize distance. If different (0), maximize distance up to margin.
    return tf.reduce_mean(y_true * tf.square(y_pred) + 
                          (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))

siamese_model.compile(loss=contrastive_loss, optimizer='adam')

# --- 3. Training ---
print("Starting Siamese Training...")
siamese_model.fit(
    [X_pairs[0], X_pairs[1]], y_pairs,
    batch_size=16,
    epochs=50,
    shuffle=True
)

base_network.save('siamese_encoder.h5')
print("✅ Success! Siamese encoder saved.")