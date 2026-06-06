import json
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model_definition import build_cnn_model
from utils.helpers import CLASS_NAMES, EMOTIONS, IMG_SIZE, create_dir_if_not_exists

BATCH_SIZE = 128
EPOCHS = 35
VALIDATION_SPLIT = 0.15
SEED = 42
LEARNING_RATE = 1e-3

DATA_DIR = "data"
MODELS_DIR = "models"
REPORTS_DIR = "reports"


def configure_tf():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("[INFO] GPU detected. TensorFlow will use it when possible.")
        except Exception as e:
            print(f"[WARNING] GPU configuration failed: {e}")
    else:
        print("[INFO] No GPU detected. Training will use CPU.")


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def create_generators(train_path):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=VALIDATION_SPLIT,
        rotation_range=10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    validation_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=VALIDATION_SPLIT,
    )

    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        subset="training",
        shuffle=True,
        seed=SEED,
    )

    validation_generator = validation_datagen.flow_from_directory(
        train_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        subset="validation",
        shuffle=False,
        seed=SEED,
    )

    return train_generator, validation_generator


def compile_model(model):
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_callbacks():
    return [
        ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, "best_model.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, "best_model.h5"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=0,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=8,
            min_delta=0.002,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        CSVLogger(os.path.join(REPORTS_DIR, "training_log.csv"), append=False),
        TerminateOnNaN(),
    ]


def save_training_plot(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Train Loss", linewidth=2)
    plt.plot(history["val_loss"], label="Val Loss", linewidth=2)
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(history["accuracy"], label="Train Accuracy", linewidth=2)
    plt.plot(history["val_accuracy"], label="Val Accuracy", linewidth=2)
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, "training_history.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def save_validation_report(model, validation_generator):
    validation_generator.reset()
    predictions = model.predict(
        validation_generator,
        steps=math.ceil(validation_generator.samples / BATCH_SIZE),
        verbose=1,
    )
    y_pred = np.argmax(predictions, axis=1)[:validation_generator.samples]
    y_true = validation_generator.classes[:len(y_pred)]

    report = classification_report(y_true, y_pred, target_names=EMOTIONS, digits=4)
    report_path = os.path.join(REPORTS_DIR, "validation_classification_report.txt")
    with open(report_path, "w") as f:
        f.write("=== Validation Classification Report ===\n")
        f.write("Validation split comes from data/train only. data/test is reserved for final evaluation.\n\n")
        f.write(report)

    print("\nValidation Classification Report:")
    print(report)
    return report_path


def main():
    print("=== Medium FER CNN Training Pipeline ===")
    create_dir_if_not_exists(MODELS_DIR)
    create_dir_if_not_exists(REPORTS_DIR)
    configure_tf()

    train_path = os.path.join(DATA_DIR, "train")
    test_path = os.path.join(DATA_DIR, "test")
    if not os.path.exists(train_path):
        print(f"[ERROR] Training folder not found: {train_path}")
        return
    if not os.path.exists(test_path):
        print(f"[ERROR] Test folder not found: {test_path}")
        return

    print("[INFO] data/test is not used during training. It is only for final evaluation.")
    train_generator, validation_generator = create_generators(train_path)
    print(f"Train samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Class mapping: {train_generator.class_indices}")

    model = compile_model(build_cnn_model())
    model.summary()

    history_obj = model.fit(
        train_generator,
        steps_per_epoch=math.ceil(train_generator.samples / BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=math.ceil(validation_generator.samples / BATCH_SIZE),
        callbacks=make_callbacks(),
        workers=4,
        use_multiprocessing=False,
        max_queue_size=16,
    )

    history = json_safe(history_obj.history)
    best_val_acc = max(history.get("val_accuracy", [0.0]))

    history_path = os.path.join(REPORTS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    config_path = os.path.join(REPORTS_DIR, "training_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "model": "FER_Medium_CNN",
            "img_size": IMG_SIZE,
            "color_mode": "grayscale",
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "validation_split": VALIDATION_SPLIT,
            "learning_rate": LEARNING_RATE,
            "class_indices": train_generator.class_indices,
            "best_validation_accuracy": best_val_acc,
        }, f, indent=4)

    best_model_path = os.path.join(MODELS_DIR, "best_model.keras")
    best_model = load_model(best_model_path) if os.path.exists(best_model_path) else model

    plot_path = save_training_plot(history)
    report_path = save_validation_report(best_model, validation_generator)

    print("\n[SUCCESS] Training completed.")
    print(f"Best validation accuracy: {best_val_acc * 100:.2f}%")
    print(f"Best model saved: {os.path.join(MODELS_DIR, 'best_model.keras')}")
    print(f"Legacy H5 copy saved: {os.path.join(MODELS_DIR, 'best_model.h5')}")
    print(f"History saved: {history_path}")
    print(f"Config saved: {config_path}")
    print(f"Training plot saved: {plot_path}")
    print(f"Validation report saved: {report_path}")


if __name__ == "__main__":
    main()
