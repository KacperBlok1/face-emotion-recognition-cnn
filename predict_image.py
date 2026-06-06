import argparse
import ctypes
import os
import sys

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from utils.helpers import EMOTION_COLORS, EMOTIONS, create_dir_if_not_exists, expand_box, preprocess_face


def find_default_model():
    for path in ("models/best_model.keras", "models/best_model.h5"):
        if os.path.exists(path):
            return path
    return "models/best_model.keras"


def get_short_path(path):
    try:
        buffer = ctypes.create_unicode_buffer(260)
        result = ctypes.windll.kernel32.GetShortPathNameW(path, buffer, 260)
        if result > 0:
            return buffer.value
    except Exception:
        pass
    return path


def load_face_detector():
    local_cascade = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml")
    cascade_path = local_cascade if os.path.exists(local_cascade) else os.path.join(
        cv2.data.haarcascades,
        "haarcascade_frontalface_default.xml",
    )

    face_cascade = cv2.CascadeClassifier(get_short_path(cascade_path))
    if face_cascade.empty():
        face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"[ERROR] Could not load Haar Cascade from: {cascade_path}")
        sys.exit(1)
    return face_cascade


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return b, g, r


def main():
    parser = argparse.ArgumentParser(description="Predict face emotions on one image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default=find_default_model(), help="Path to trained model")
    parser.add_argument("--output_dir", type=str, default="reports", help="Directory for output image")
    parser.add_argument("--margin", type=float, default=0.15, help="Face crop margin")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        print("Run python train.py first.")
        sys.exit(1)

    image = cv2.imread(args.image)
    if image is None:
        print(f"[ERROR] Could not read image: {args.image}")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    model = load_model(args.model)

    output_img = image.copy()
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = load_face_detector()

    faces = detector.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(35, 35),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    print(f"Detected {len(faces)} face(s).")
    if len(faces) == 0:
        return

    for i, (x, y, w, h) in enumerate(faces, start=1):
        x1, y1, x2, y2 = expand_box(x, y, w, h, image.shape, margin=args.margin)
        face_roi = image[y1:y2, x1:x2]
        preprocessed = preprocess_face(face_roi)
        if preprocessed is None:
            print(f"[WARNING] Skipping face #{i}; preprocessing failed.")
            continue

        preds = model.predict(preprocessed, verbose=0)[0]
        max_idx = int(np.argmax(preds))
        label = EMOTIONS[max_idx]
        confidence = float(preds[max_idx])

        print(f"\nFace #{i}: {label} ({confidence * 100:.2f}%)")
        for idx, emotion in enumerate(EMOTIONS):
            print(f" - {emotion}: {float(preds[idx]) * 100:.2f}%")

        color = hex_to_bgr(EMOTION_COLORS.get(label, "#10b981"))
        text = f"{label} {confidence * 100:.1f}%"
        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.65
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        label_y = max(0, y1 - th - 12)
        cv2.rectangle(output_img, (x1, label_y), (x1 + tw + 8, y1), color, cv2.FILLED)
        cv2.putText(output_img, text, (x1 + 4, y1 - 6), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    create_dir_if_not_exists(args.output_dir)
    name, ext = os.path.splitext(os.path.basename(args.image))
    output_path = os.path.join(args.output_dir, f"{name}_prediction{ext}")
    cv2.imwrite(output_path, output_img)
    print(f"\n[SUCCESS] Saved output image: {output_path}")


if __name__ == "__main__":
    main()
