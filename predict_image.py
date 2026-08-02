import os
import sys
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

from model_utils import MODEL_PKL_PATH, load_model_from_pickle


def classify_image(image_obj, model):
    image_rgb = image_obj.convert("RGB")
    image_resized = image_rgb.resize((224, 224))
    image_array = keras_image.img_to_array(image_resized)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    preds = model.predict(image_array)
    decoded = decode_predictions(preds, top=1)[0][0]
    predicted_class = decoded[1].replace("_", " ")
    confidence_score = float(decoded[2] * 100)
    return predicted_class, confidence_score


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_image.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    model = load_model_from_pickle(MODEL_PKL_PATH)
    image_obj = Image.open(image_path)

    predicted_class, confidence_score = classify_image(image_obj, model)
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence_score:.2f}%")


if __name__ == "__main__":
    main()
