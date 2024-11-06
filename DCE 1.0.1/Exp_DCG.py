import cv2
import mediapipe as mp
import csv

import tensorflow as tf
from keras.layers import DepthwiseConv2D
from keras.models import load_model
#import cv2
import numpy as np

# Initialize MediaPipe hands, face detection, and drawing utilities
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Start the camera feed
cap = cv2.VideoCapture(0)

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
    model = load_model("DCG 0-4-1.h5", custom_objects=custom_objects, compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Load the labels
class_names = open("labels.txt", "r").readlines()
camera = cv2.VideoCapture(0)


# Use MediaPipe Hands and Face Detection with specific parameters
with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands, \
     mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detection:

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        # Get frame dimensions (height, width)
        height, width, _ = image.shape
        # Determine the size of the largest square
        square_size = min(height, width)

        # Flip the frame horizontally for a mirror effect
        flip_image = cv2.flip(image, 1)
        # Convert the frame to RGB as MediaPipe expects RGB input
        rgb_frame = cv2.cvtColor(flip_image, cv2.COLOR_BGR2RGB)

        # Perform hand and face detection
        hand_results = hands.process(rgb_frame)
        face_results = face_detection.process(rgb_frame)

        # Initialize empty lists for left, right hand points, and head position
        left_list, right_list = [], []
        head_position = None

        # Get frame dimensions (height, width)
        height, width, _ = image.shape

        # Determine the size of the largest square
        square_size = min(height, width)

        # Calculate the top-left corner of the square crop area (center crop)
        x_center = width // 2
        y_center = height // 2
        x_start = x_center - (square_size // 2)
        y_start = y_center - (square_size // 2)

        # If hands are detected
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # Check if it's a left or right hand
                hand_label = hand_results.multi_handedness[idx].classification[0].label
                hand_points = []

                # Collect all normalized coordinates for the current hand
                for landmark in hand_landmarks.landmark:
                    hand_points.append((landmark.x, landmark.y))

                # Assign points to the respective list based on hand label
                if hand_label == 'Left':
                    left_list = hand_points
                else:
                    right_list = hand_points

                # Draw hand landmarks and bounding box on the frame
                #mp_drawing.draw_landmarks(flip_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get coordinates for bounding box around the hand
                x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * image.shape[1])
                y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * image.shape[0])
                x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * image.shape[1])
                y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * image.shape[0])

                # Draw the bounding box around the hand
                cv2.rectangle(flip_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Draw a black box at the top of the bounding box with "sign" label
                box_position = (x_min, y_min - 30)
                cv2.rectangle(flip_image, box_position, (x_min + 150, y_min), (0, 0, 0), -1)

                try:
                    conf = str(np.round(confidence_score * 100))[:-2]
                    result_display = str(class_name[2:] + conf + "%")
                except:
                    None
                try:
                    cv2.putText(flip_image, result_display, (x_min + 5, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255),
                                1)
                except:
                    None

        # Detect and display face bounding box if a face is detected
        if face_results.detections:
            for detection in face_results.detections:
                # Extract bounding box information for head position
                bbox = detection.location_data.relative_bounding_box
                x_min = int(bbox.xmin * image.shape[1])
                y_min = int(bbox.ymin * image.shape[0])
                width = int(bbox.width * image.shape[1])
                height = int(bbox.height * image.shape[0])
                x_max = x_min + width
                y_max = y_min + height

                # Approximate head position as the center of the face bounding box
                head_position = ((x_min + x_max) // 2, (y_min + y_max) // 2)

                # Draw the bounding box around the head
                cv2.rectangle(flip_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                # Draw a black box at the top of the bounding box with "head" label
                box_position = (x_min, y_min - 30)
                cv2.rectangle(flip_image, box_position, (x_min + 50, y_min), (0, 0, 0), -1)
                cv2.putText(flip_image, "head", (x_min + 5, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                            1)

        # Crop the largest square region from the center
        image = image[y_start:y_start + square_size, x_start:x_start + square_size]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #recolour
        # Resize the raw image into (224-height,224-width) pixels
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
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
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

        # Display the output
        cv2.imshow("Hand and Head Detection", flip_image)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF

        # If 'p' is pressed, print the normalized coordinates for left and right hands and head
        if key == ord('p'):
            print("Left Hand Normalized Coordinates:", left_list)
            print("Right Hand Normalized Coordinates:", right_list)

            # Print normalized head position
            if head_position:
                normalized_head_x = head_position[0] / frame.shape[1]
                normalized_head_y = head_position[1] / frame.shape[0]
                normalized_head = [(normalized_head_x,normalized_head_y)]
                print("Head Normalized Position:", normalized_head)
            else:
                print("Head Position: Not Detected")
            print("major list")
            major_list = left_list+right_list+normalized_head
            print(major_list)

            # Specify the CSV file name
            csv_file = 'output.csv'

            # Create or open the CSV file
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)

                # Write header for the first column
                header = ["A"]
                # Start with an empty data structure for the rest of the columns
                additional_data = []
                # Get label from the user
                label = input("Enter a label for the new column (or 'exit' to finish): ")
                if label.lower() == 'exit':
                     break
                # Append the label to the header
                header.append(label)

                # Gather data for this column
                column_data = []
                for i, (a, b) in enumerate(major_list):
                    column_data.append(a)
                    column_data.append(b)

                # Append the collected column data
                additional_data.append(column_data)

                # Write the header to the CSV
                writer.writerow(header)

                # Write the data
                for row in zip(*additional_data):  # Unpack the lists to create rows
                    writer.writerow(row)

        # Break the loop on 'Esc' key press
        if key == 27:  # ASCII for Esc key is 27
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
