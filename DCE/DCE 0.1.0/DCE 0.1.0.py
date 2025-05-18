import tensorflow as tf
from keras.layers import DepthwiseConv2D
from keras.models import load_model
import cv2
import numpy as np
import pyttsx3
import threading
import speech_recognition as sr

# Define a custom DepthwiseConv2D class without the groups parameter
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        # Remove the 'groups' parameter if it exists
        if 'groups' in kwargs:
            del kwargs['groups']  # Remove the groups parameter
        super().__init__(**kwargs)

# Create a dictionary of custom objects to pass to the load_model function
custom_objects = {
    'DepthwiseConv2D': CustomDepthwiseConv2D,
}

# Load the model with the custom object
try:
    model = load_model("keras_model.h5", custom_objects=custom_objects, compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
word_said = []


# Preprocessing function for input frames
def load_and_preprocess_image(image_path):
    # Load the image from the specified file path
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    square_size = min(height, width)
    x_center = width // 2
    y_center = height // 2
    x_start = x_center - (square_size // 2)
    y_start = y_center - (square_size // 2)
    cropped_frame = image[y_start:y_start + square_size, x_start:x_start + square_size]

    # Check if the image was loaded successfully
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Resize the image to the expected input size of the model (224x224)
    image = cv2.resize(cropped_frame, (224, 224), interpolation=cv2.INTER_AREA)

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize the image data
    image_rgb = np.asarray(image_rgb, dtype=np.float32) / 255.0

    # Reshape the image to match the model's input shape
    input_frame = image_rgb.reshape(1, 224, 224, 3)

    return input_frame

while True:
    # Specify the image file path
    image_file_path = input("enter file path:")

    # Load and preprocess the image
    input_frame = load_and_preprocess_image(image_file_path)

    # Make predictions
    predictions = model.predict(input_frame)
    # Get the class index and confidence score
    index = np.argmax(predictions)
    confidence_score = predictions[0][index]
    # Load the labels
    class_names = open("labels.txt", "r").readlines()

    # Print prediction and confidence score
    print("Class:", class_names[index].strip())
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
