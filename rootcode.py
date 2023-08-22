import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
TUMOR_CLASSES = ["category1_tumor", "category2_tumor", "category3_tumor", "no_tumor"]
TEST_SIZE = 0.4
DIRECTORY = "../Datathon-Dataset"


def main():

    # Get image arrays and labels for all image files
    images, labels = load_data(DIRECTORY)

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 2:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(base_path):

    images = []
    labels = []

    for i, tumor_class in enumerate(TUMOR_CLASSES):
        class_path = os.path.join(base_path, tumor_class)

        for filename in os.listdir(class_path):
            if filename.endswith(".jpg"):
                img = cv2.imread(os.path.join(class_path, filename))
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                images.append(img)
                labels.append(i)
    return images, labels


def get_model():
    layers = [
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Add hidden layers with dropout
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Output layer
        tf.keras.layers.Dense(4, activation="softmax"),
    ]
    model = tf.keras.models.Sequential(layers)
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


if __name__ == "__main__":
    main()
