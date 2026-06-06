from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

from utils.helpers import EMOTIONS, IMG_SIZE


def conv_block(model, filters, dropout_rate):
    model.add(Conv2D(filters, (3, 3), padding="same", kernel_regularizer=l2(1e-4), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(filters, (3, 3), padding="same", kernel_regularizer=l2(1e-4), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))


def build_cnn_model():
    """
    Build a medium-complexity CNN for FER-2013.

    It is intentionally not a heavy transfer-learning model. The goal is a project
    that trains quickly, is easy to explain, and still uses solid CNN practices:
    convolution blocks, batch normalization, dropout, and global average pooling.
    """
    model = Sequential(name="FER_Medium_CNN")
    model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 1)))

    conv_block(model, 32, 0.20)
    conv_block(model, 64, 0.25)
    conv_block(model, 128, 0.30)

    model.add(Conv2D(192, (3, 3), padding="same", kernel_regularizer=l2(1e-4), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(128, activation="relu", kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.45))
    model.add(Dense(len(EMOTIONS), activation="softmax", name="emotion_predictions"))

    return model


if __name__ == "__main__":
    model = build_cnn_model()
    model.summary()
