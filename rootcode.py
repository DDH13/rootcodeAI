import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import time
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
TUMOR_CLASSES = ["category1_tumor", "category2_tumor", "category3_tumor", "no_tumor"]
TEST_SIZE = 0.4
DIRECTORY = "../Datathon-Dataset"


def main():
    # Seed for reproducibility
    tf.random.set_seed(123)

    # Get image arrays and labels for all image files
    images, labels = load_data(DIRECTORY)

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    early_stopping = EarlyStopping(monitor='val_loss', patience=2,restore_best_weights=True, verbose=2)
    # Fit model on training data
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=EPOCHS, callbacks=[early_stopping], validation_split=0.2,
                        validation_data=(x_test, y_test), verbose=2)
    end_time = time.time()

    # Evaluate neural network performance
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)

    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions, axis=1)
    y_test = np.argmax(y_test, axis=1)
    
    print("Time taken: ", end_time - start_time, " seconds")
    print(f"\nLoss: {loss}")
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred, average='weighted'))
    print("Precision: ", precision_score(y_test, y_pred, average='weighted'))
    print("F1 Score: ", f1_score(y_test, y_pred, average='weighted'))

    with open('logs.txt', 'a') as f:
        #write model summary and evaluation metrics to file
        f.write("\n\n\nModel Summary: \n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\nTime taken: " + str(end_time - start_time) + " seconds" )
        f.write("\nLoss: " + str(loss))
        f.write("\nAccuracy: " + str(accuracy))
        f.write("\nRecall: " + str(recall_score(y_test, y_pred, average='weighted')))
        f.write("\nPrecision: " + str(precision_score(y_test, y_pred, average='weighted')))
        f.write("\nF1 Score: " + str(f1_score(y_test, y_pred, average='weighted')))
        f.write("_________________________________________________________________")


    # Save model to file
    if len(sys.argv) == 2:
        filename = sys.argv[1]
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
                img = enhance_image(img)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                images.append(img)
                labels.append(i)
    return images, labels


def enhance_image(img):
    img = Image.fromarray(np.uint8(img))
    img = ImageEnhance.Brightness(img).enhance(1.5)
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    img = np.array(img) / 255.0
    return img


def get_model():
    layers = [
        tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding='same'),

        tf.keras.layers.Conv2D(256, (3, 3), activation="relu",padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding='same'),

        tf.keras.layers.Conv2D(512, (3, 3), activation="relu",padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding='same'),

        tf.keras.layers.Conv2D(512, (3, 3), activation="relu",padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding='same'),

        tf.keras.layers.Conv2D(512, (3, 3), activation="relu",padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2),padding='same'),

        # Flatten units
        tf.keras.layers.Flatten(),

        # Add hidden layers with dropout
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.2),

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
