import os

import numpy as np

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
CLASS_NAMES = [emotion.lower() for emotion in EMOTIONS]

EMOTION_COLORS = {
    "Angry": "#ef4444",
    "Disgust": "#8b5cf6",
    "Fear": "#374151",
    "Happy": "#10b981",
    "Neutral": "#6b7280",
    "Sad": "#3b82f6",
    "Surprise": "#f59e0b",
}

# Medium-complexity project setting: fast FER-style grayscale CNN.
IMG_SIZE = 48
MODEL_COLOR_MODE = "grayscale"


def preprocess_face(face_image):
    """
    Preprocess a detected face ROI for the medium CNN:
    - accepts grayscale or BGR image arrays,
    - converts to grayscale,
    - resizes to 48x48,
    - normalizes pixels to [0, 1],
    - returns shape (1, 48, 48, 1).
    """
    try:
        import cv2

        if face_image is None or face_image.size == 0:
            return None

        if len(face_image.shape) == 2:
            gray = face_image
        else:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        normalized = resized.astype("float32") / 255.0
        return np.expand_dims(np.expand_dims(normalized, axis=0), axis=-1)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None


def expand_box(x, y, w, h, image_shape, margin=0.15):
    """Expand a face bounding box slightly so the model gets facial context."""
    img_h, img_w = image_shape[:2]
    pad_x = int(w * margin)
    pad_y = int(h * margin)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)
    return x1, y1, x2, y2


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
