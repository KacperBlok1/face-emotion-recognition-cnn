import math
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils.helpers import CLASS_NAMES, EMOTIONS, IMG_SIZE, create_dir_if_not_exists

BATCH_SIZE = 128
DATA_DIR = "data"
MODELS_DIR = "models"
REPORTS_DIR = "reports"


def find_model_path():
    for filename in ("best_model.keras", "best_model.h5"):
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            return path
    return None


def main():
    print("=== FER CNN Final Test Evaluation ===")
    create_dir_if_not_exists(REPORTS_DIR)

    model_path = find_model_path()
    if model_path is None:
        print(f"[ERROR] No trained model found in '{MODELS_DIR}'. Run python train.py first.")
        return

    test_path = os.path.join(DATA_DIR, "test")
    if not os.path.exists(test_path):
        print(f"[ERROR] Test folder not found: {test_path}")
        return

    print(f"Loading model: {model_path}")
    model = load_model(model_path)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False,
    )

    loss, eval_accuracy = model.evaluate(
        test_generator,
        steps=math.ceil(test_generator.samples / BATCH_SIZE),
        verbose=1,
    )

    test_generator.reset()
    predictions = model.predict(
        test_generator,
        steps=math.ceil(test_generator.samples / BATCH_SIZE),
        verbose=1,
    )
    y_pred = np.argmax(predictions, axis=1)[:test_generator.samples]
    y_true = test_generator.classes[:len(y_pred)]
    accuracy = float(np.mean(y_pred == y_true))

    print("\n=============================================")
    print(f"FINAL TEST LOSS: {float(loss):.4f}")
    print(f"FINAL TEST ACCURACY: {accuracy * 100:.2f}%")
    print("=============================================\n")

    report = classification_report(y_true, y_pred, target_names=EMOTIONS, digits=4)
    print(report)

    report_path = os.path.join(REPORTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("=== FER CNN Final Test Classification Report ===\n")
        f.write("data/test is used only here, not during training.\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Input: {IMG_SIZE}x{IMG_SIZE} grayscale\n")
        f.write(f"Final Test Loss: {float(loss):.4f}\n")
        f.write(f"Final Test Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100

    plt.figure(figsize=(9, 8))
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=EMOTIONS,
        yticklabels=EMOTIONS,
        cbar_kws={"label": "Class accuracy (%)"},
    )
    plt.title("Final Test Confusion Matrix (%)", fontsize=14, pad=15)
    plt.xlabel("Predicted Emotion")
    plt.ylabel("True Emotion")
    plt.tight_layout()

    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    print(f"Saved report: {report_path}")
    print(f"Saved confusion matrix: {cm_path}")
    print("[SUCCESS] Evaluation complete.")


if __name__ == "__main__":
    main()
