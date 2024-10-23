import tensorflow as tf
from keras.layers import DepthwiseConv2D
from keras.models import load_model
import cv2
import numpy as np

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
    model = load_model("DCG 0-3-1.h5", custom_objects=custom_objects, compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Load the labels
class_names = open("labels.txt", "r").readlines()
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Get frame dimensions (height, width)
    height, width, _ = image.shape

    # Determine the size of the largest square
    square_size = min(height, width)

    # Calculate the top-left corner of the square crop area (center crop)
    x_center = width // 2
    y_center = height // 2
    x_start = x_center - (square_size // 2)
    y_start = y_center - (square_size // 2)

    # Crop the largest square region from the center
    image = image[y_start:y_start + square_size, x_start:x_start + square_size]

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], "Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()