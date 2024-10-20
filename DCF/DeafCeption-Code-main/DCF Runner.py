import tensorflow as tf
from keras.layers import DepthwiseConv2D
from keras.models import load_model
import cv2
import numpy as np
import pyttsx3
import threading
import speech_recognition as sr
import os

def speak_non_blocking(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Define a custom DepthwiseConv2D class without the groups parameter
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        # Remove the 'groups' parameter if it exists
        if 'groups' in kwargs:
            del kwargs['groups']  # Remove the groups parameter
        super().__init__(**kwargs)

# Initialize the TTS engine
engine = pyttsx3.init()
# Set properties for voice (optional)
engine.setProperty('rate', 150)  # Speed (words per minute)
engine.setProperty('volume', 1)  # Volume (0.0 to 1.0)


# Create a dictionary of custom objects to pass to the load_model function
custom_objects = {
    'DepthwiseConv2D': CustomDepthwiseConv2D,
}

# Load the model with the custom object
model_folder = os.path.join(os.getcwd(),)  # Ensure that it dynamically picks the current directory
model_path = os.path.join(model_folder, "keras_Model.h5")
try:
    model = load_model(model_path, custom_objects=custom_objects, compile=False)
    print(f"Model loaded successfully from: {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")

word_said = []

# Preprocessing function for input frames
def preprocess_frame(frame):
    # Resize or normalize frame as needed for your model
    frame = cv2.resize(frame, (224, 224))  # Example resize, adjust as needed
    frame = frame.astype('float32') / 255.0  # Normalize to [0, 1]
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame


# Start capturing video from the camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
counter = 0
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Check if the frame is completely black (all pixels zero)
    if np.sum(frame) == 0:
        print("Warning: Captured frame is completely black.")
    else:
        input_frame = preprocess_frame(frame)
        if model is not None:
            # Use the model for prediction
            predictions = model.predict(input_frame)
            predicted_class = np.argmax(predictions)

            if predicted_class == 0:
                text = 'a'
            elif predicted_class == 1:
                text = 'b'
            elif predicted_class == 2:
                text = 'c'
            elif predicted_class == 3:
                text = 'd'

            cv2.putText(frame, f'Predicted Sign: {text}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            if counter == 20:
                if predicted_class == 0:
                    thread = threading.Thread(target=speak_non_blocking, args=('a',))
                    thread.start()
                elif predicted_class == 1:
                    thread = threading.Thread(target=speak_non_blocking, args=('b',))
                    thread.start()
                elif predicted_class == 2:
                    thread = threading.Thread(target=speak_non_blocking, args=('c',))
                    thread.start()
                elif predicted_class == 3:
                    thread = threading.Thread(target=speak_non_blocking, args=('d',))
                    thread.start()

            else:
                None

            prev_label = predicted_class
        else:
            print("Model is not defined. Check if the model is loaded properly.")

    cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('s'):
        if counter == 20:
            word_said.append(predicted_class)
        else:
            None
    if cv2.waitKey(1) & 0xFF == ord('s'):
        thread = threading.Thread(target=speak_non_blocking, args=(word_said,))
        thread.start()
    if counter == 20:
        counter = 0
        counter +=1
    else:
        counter +=1

cap.release()
cv2.destroyAllWindows()

# Initialize the recognizer
recognizer = sr.Recognizer()

# Capture audio from the microphone
with sr.Microphone() as source:
    print("Please say something:")
    audio = recognizer.listen(source)

    try:
        # Use Google's speech recognition to convert audio to text
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
    except sr.UnknownValueError:
        print("Couldn't get what you said")
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition service.")
