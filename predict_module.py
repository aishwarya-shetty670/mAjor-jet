# predict_module.py
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Adjust if your model file is somewhere else
MODEL_PATH = "severity_final.keras"
IMG_SIZE = (224, 224)

# IMPORTANT: class names in the same order as training
# Check train_generator.class_indices output to confirm
# Example: {'moderate': 0, 'normal': 1, 'severe': 2}
CLASS_NAMES = ["moderate", "normal", "severe"]

print("Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded from", MODEL_PATH)


def load_image_any(src: str) -> Image.Image | None:
    """
    Load an image from a local path or an HTTP/HTTPS URL.
    Returns a PIL.Image object or None on error.
    """
    try:
        if src.startswith("http://") or src.startswith("https://"):
            resp = requests.get(src, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
        else:
            img = Image.open(src).convert("RGB")
        return img
    except Exception as e:
        print("❌ Error loading image:", e)
        return None


def predict_severity(img: Image.Image):
    """
    Takes a PIL image, returns (label, confidence, probabilities array).
    """
    arr = img.resize(IMG_SIZE)
    arr = img_to_array(arr)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

    probs = model.predict(arr)[0]   # shape (3,)
    idx = np.argmax(probs)
    label = CLASS_NAMES[idx]
    conf = float(probs[idx])
    return label, conf, probs
