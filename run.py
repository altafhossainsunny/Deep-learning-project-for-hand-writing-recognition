# run.py
import os, cv2, numpy as np, pandas as pd, joblib, tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

TEST_PATH = "test"
IMG_SIZE = 64

model = tf.keras.models.load_model("writer_model.keras")
le = joblib.load("labels.pkl")
writers = le.classes_

y_true, y_pred, y_probs = [], [], []

print("🔮 Scanning Test Images...")
for fname in sorted(os.listdir(TEST_PATH)):
    if not fname.lower().endswith(".png"): continue
    img = cv2.imread(os.path.join(TEST_PATH, fname), cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    patches = []
    h, w = img.shape
    for i in range(0, h-IMG_SIZE, 32):
        for j in range(0, w-IMG_SIZE, 32):
            p = img[i:i+IMG_SIZE, j:j+IMG_SIZE]
            p = cv2.cvtColor(p, cv2.COLOR_GRAY2RGB)
            patches.append(p / 255.0)

    preds = model.predict(np.array(patches), verbose=0)
    avg_prob = np.mean(preds, axis=0) # Majority vote from all parts of the page

    y_true.append(fname[:2])
    y_pred.append(le.inverse_transform([np.argmax(avg_prob)])[0])
    y_probs.append(avg_prob)

# Metrics
y_true_idx = le.transform(y_true)
y_pred_idx = le.transform(y_pred)
acc = accuracy_score(y_true_idx, y_pred_idx)
y_bin = label_binarize(y_true_idx, classes=range(len(writers)))
roc = roc_auc_score(y_bin, np.array(y_probs), average="macro", multi_class="ovr")

print(f"\n🎯 Accuracy : {acc:.4f}\n📊 ROC-AUC  : {roc:.4f}")
pd.DataFrame({"filename": sorted(os.listdir(TEST_PATH)), "actual": y_true, "predicted": y_pred}).to_csv("result.csv", index=False)