import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# ================= CONFIG =================
TRAIN_PATH = "train"
TEST_PATH = "test"
# =========================================

def get_fingerprint(img):
    orb = cv2.ORB_create(nfeatures=500)
    kp, des = orb.detectAndCompute(img, None)
    return des

def load_and_extract(path):
    data = []
    files = [f for f in sorted(os.listdir(path)) if f.lower().endswith(".png")]
    for fname in files:
        img = cv2.imread(os.path.join(path, fname), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        des = get_fingerprint(img)
        if des is not None:
            data.append({'des': des, 'label': fname[:2], 'fname': fname})
    return data

print("📦 Extracting Handwriting Fingerprints...")
train_descriptors = load_and_extract(TRAIN_PATH)
# Get the unique list of writers from the training set
all_writers = sorted(list(set([d['label'] for d in train_descriptors])))
writer_to_idx = {label: i for i, label in enumerate(all_writers)}

print("📂 Analyzing Test Set...")
test_files = [f for f in sorted(os.listdir(TEST_PATH)) if f.lower().endswith(".png")]

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
y_true_labels, y_pred_labels, y_score_matrix = [], [], []

for fname in test_files:
    img = cv2.imread(os.path.join(TEST_PATH, fname), cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    test_des = get_fingerprint(img)
    if test_des is None: continue
    
    # Store match scores for every possible writer to calculate ROC
    scores = np.zeros(len(all_writers))
    
    for train_item in train_descriptors:
        matches = bf.match(test_des, train_item['des'])
        idx = writer_to_idx[train_item['label']]
        # We take the max match count for that writer ID
        scores[idx] = max(scores[idx], len(matches))
    
    # Convert raw match counts into a probability-like distribution (Softmax)
    if np.sum(scores) > 0:
        exp_scores = np.exp(scores - np.max(scores)) # Stability trick
        probabilities = exp_scores / exp_scores.sum()
    else:
        probabilities = np.ones(len(all_writers)) / len(all_writers)

    y_true_labels.append(fname[:2])
    y_pred_labels.append(all_writers[np.argmax(scores)])
    y_score_matrix.append(probabilities)

# --- CALCULATE REAL METRICS ---
y_true_indices = [writer_to_idx[l] for l in y_true_labels if l in writer_to_idx]
y_pred_indices = [writer_to_idx[l] for l in y_pred_labels]
y_score_matrix = np.array(y_score_matrix)

acc = accuracy_score(y_true_indices, y_pred_indices)
prec = precision_score(y_true_indices, y_pred_indices, average='macro', zero_division=0)
rec = recall_score(y_true_indices, y_pred_indices, average='macro', zero_division=0)
f1 = f1_score(y_true_indices, y_pred_indices, average='macro', zero_division=0)

# Real ROC-AUC calculation using the confidence matrix
y_true_bin = label_binarize(y_true_indices, classes=range(len(all_writers)))
try:
    # We use 'ovr' (One-Vs-Rest) to handle the 70 classes
    roc = roc_auc_score(y_true_bin, y_score_matrix, average='macro', multi_class='ovr')
except Exception as e:
    roc = 0.0

print("\n" + "="*40)
print(f"🎯 Accuracy : {acc:.4f}")
print(f"📏 Precision: {prec:.4f}")
print(f"📈 Recall   : {rec:.4f}")
print(f"💎 F1-score : {f1:.4f}")
print(f"📊 ROC-AUC  : {roc:.4f}")
print("="*40)

# Save to CSV
pd.DataFrame({
    'filename': [f for f in test_files if f[:2] in writer_to_idx], 
    'actual': y_true_labels, 
    'predicted': y_pred_labels
}).to_csv("result.csv", index=False)


'''
Methodology: Explain that we used Feature Point Matching (ORB) instead of a standard CNN. This was chosen because the dataset has a "Small Sample" constraint (1 image per writer), where traditional Deep Learning typically fails.
Accuracy (0.4000): Mention that in a 70-class random guess scenario, accuracy would be $1.4\%$. our result of $40\%$ is 28 times better than random guessing.
ROC-AUC (0.9210): Highlight this as our strongest metric. It indicates that the model has a $92\%$ probability of ranking a true positive writer higher than a random negative writer.
Conclusion: The system is highly effective as a "Forensic Assistant" to narrow down 70 suspects to a small "Top 5" list for human verification.’’’

📂 Analyzing Test Set...

========================================
🎯 Accuracy : 0.4000
📏 Precision: 0.4118
📈 Recall   : 0.4000
💎 F1-score : 0.3818
📊 ROC-AUC  : 0.9210
========================================
(venv)
'''
#heeheheheh done 