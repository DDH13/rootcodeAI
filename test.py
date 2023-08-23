import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

TUMOR_CLASSES = ["category1_tumor", "category2_tumor", "category3_tumor", "no_tumor"]
IMG_WIDTH = 80
IMG_HEIGHT = 80

def load_data_for_testing(base_path):
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <model_filename>")
        sys.exit(1)

    model_filename = sys.argv[1]
    if not os.path.exists(model_filename):
        print(f"Model file '{model_filename}' not found.")
        sys.exit(1)

    model = tf.keras.models.load_model(model_filename)

    # Load test data
    test_images, test_labels = load_data_for_testing("../Datathon-Dataset")

    # Preprocess test data
    test_images = np.array(test_images)
    test_labels = tf.keras.utils.to_categorical(test_labels)

    # Evaluate model performance
    loss, accuracy = model.evaluate(test_images, test_labels, verbose=2)

    predictions = model.predict(test_images)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(test_labels, axis=1)

    print(f"Model Evaluation - Loss: {loss}, Accuracy: {accuracy}")
    print("Accuracy: ", accuracy_score(y_true, y_pred))
    print("Recall: ", recall_score(y_true, y_pred, average='weighted'))
    print("Precision: ", precision_score(y_true, y_pred, average='weighted'))
    print("F1 Score: ", f1_score(y_true, y_pred, average='weighted'))
