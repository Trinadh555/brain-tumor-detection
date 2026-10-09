"""
Predict the tumor type for one MRI image.

Usage:
    python predict_brain_tumor.py path/to/image.jpg
or just run it without arguments (for example with the Run button) and it
will ask you for the image path.

Needs brain_tumor_model.keras and class_names.json (both created by
train_brain_tumor.py). The class order is read from class_names.json,
so it always matches the order used during training.
"""
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow info logs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "brain_tumor_model.keras"

with open("class_names.json") as f:
    cfg = json.load(f)
classes = cfg["class_names"]
img_size = cfg["img_size"]

model = load_model(MODEL_PATH, compile=False)


def predict(img_path):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx]) * 100


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        path = sys.argv[1]
    else:
        path = input("Testing/meningioma/Te-aug-me_2.jpg ").strip().strip('"')
    if not os.path.isfile(path):
        sys.exit(f"File not found: {path}")
    label, confidence = predict(path)
    if label == "notumor":
        print(f"No Tumor Detected (confidence {confidence:.1f}%)")
    else:
        print("Tumor Detected")
        print(f"Tumor Type: {label.capitalize()} (confidence {confidence:.1f}%)")