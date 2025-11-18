import os

predict_script_content = """
import tensorflow as tf
import numpy as np
import cv2
import os
import sys

# Define the image size used during training
IMG_SIZE = (128, 128)

# Define the path to the saved model
MODEL_PATH = 'brain_tumor_detection/model/saved_model/model.h5'

# Class mapping (must be the same as during training)
CLASS_NAMES = {
    0: 'glioma',
    1: 'meningioma',
    2: 'notumor',
    3: 'pituitary'
}

def preprocess_single_image(image_path):
    '''Loads and preprocesses a single image for prediction.'''
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    return img

def predict_image_class(model, preprocessed_img):
    '''Makes a prediction on a preprocessed image and returns the class name.'''
    if preprocessed_img is None:
        return "Preprocessing failed"

    predictions = model.predict(preprocessed_img)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class_name = CLASS_NAMES.get(predicted_class_idx, "Unknown")
    return predicted_class_name

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    image_file_path = sys.argv[1]

    print(f"Loading model from: {MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print(f"Preprocessing image: {image_file_path}")
    processed_image = preprocess_single_image(image_file_path)

    if processed_image is not None:
        print("Making prediction...")
        predicted_label = predict_image_class(model, processed_image)
        print(f"\nPredicted class for {image_file_path}: {predicted_label}")
    else:
        print(f"Could not process image: {image_file_path}")
"""

# Define the directory and file path for the script
script_dir = "brain_tumor_detection/model"
script_path = os.path.join(script_dir, "predict.py")

# Create the directory if it doesn't exist
os.makedirs(script_dir, exist_ok=True)

# Write the content to the file
with open(script_path, "w") as f:
    f.write(predict_script_content)

print(f"`predict.py` script created successfully at: {script_path}")
