import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LeakyReLU
import cv2
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
import numpy as np
import time
import csv
from tensorflow.keras.models import Sequential

# Initialize the encoder
label_encoder = LabelEncoder()

# Load CSV data
data = pd.read_csv("points_web_demo.csv")

# Split features and labels
X = data.iloc[:, :-1].values  # First 86 columns
y = data.iloc[:, -1].values   # Last column
# Encode the labels
y = label_encoder.fit_transform(data.iloc[:, -1].values)  # Convert strings to integers
y = to_categorical(y)  # Convert labels to one-hot encoding

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Data augmentation function
def augment_data(X, jitter_std=0.01, scale_range=(0.9, 1.1), shift_range=(-0.02, 0.02)):
    X_augmented = X.copy()

    # Jittering: Add small Gaussian noise
    noise = np.random.normal(0, jitter_std, X.shape)
    X_augmented += noise

    # Scaling: Apply a random scale factor
    scale_factor = np.random.uniform(scale_range[0], scale_range[1], (X.shape[0], 1))
    X_augmented *= scale_factor

    # Shifting: Add a small random shift
    shift_values = np.random.uniform(shift_range[0], shift_range[1], (X.shape[0], X.shape[1] // 2, 1))
    X_augmented[:, ::2] += shift_values[:, :, 0]  # Apply to x-coordinates
    X_augmented[:, 1::2] += shift_values[:, :, 0]  # Apply to y-coordinates

    return X_augmented


# Apply augmentation to training data
X_train_augmented = augment_data(X_train)
X_train_augmented2 = augment_data(X_train, jitter_std=0.009, scale_range=(0.85, 1.05),shift_range=(-0.01,0.01))

# (Optional) Combine with original data to make the dataset larger
X_train_combined = np.vstack([X_train, X_train_augmented,X_train_augmented2])
y_train_combined = np.vstack([y_train, y_train, y_train])

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU

# Update model for multiclass classification
# Heart of the Model
input_shape = (86,)
num_classes = y.shape[1]

inputs = Input(shape=input_shape)

x = Dense(256)(inputs)
x = LeakyReLU(alpha=0.1)(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(128)(x)
x = LeakyReLU(alpha=0.1)(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(64)(x)
x = LeakyReLU(alpha=0.1)(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Dense(32)(x)
x = LeakyReLU(alpha=0.1)(x)

outputs = Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(X_train_combined, y_train_combined, epochs=30  , batch_size=16, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)

model.summary()
print(f"Test Accuracy: {accuracy}")

import matplotlib.pyplot as plt

# Extract training history data
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

#saving model
model.save("DCGX-D.keras")

#----------------Generating Heatmap Confusion Matrix-------------------------------------------------------------
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Predict labels
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
unique_labels = np.unique(y_true)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the heatmap
plt.figure(figsize=(16, 14))
cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='coolwarm',
            xticklabels=unique_labels, yticklabels=unique_labels,
            annot_kws={"size": 6})

plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.xticks(ticks=np.arange(0, len(unique_labels), 5), labels=unique_labels[::5], rotation=90)
plt.yticks(ticks=np.arange(0, len(unique_labels), 5), labels=unique_labels[::5], rotation=0)
plt.title("Confusion Matrix Heatmap ")
plt.show()

#------------------------Mediapipe-----------------------------------------------------------------------------------

# Initialize MediaPipe hands, face detection, and drawing utilities
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Start the camera feed
cap = cv2.VideoCapture(0)

# Initialize FPS counter variables
prev_time = time.time()  # time at the start
frame_count = 0
fps = 0

detect_sentence = False
ASL_sentence = ['']
counter = 0
prev_label = ' '
letter_mode = False
recent_labels = []
immediate_speak = False
prev_spoken = ''
disp_label = '  No_symbol  '

#-----Gathering Labels---------------------------------------------------------------------------------------------

csv_file_path = 'points_web_demo.csv'

# Initialize a set to store unique labels
unique_labels = set()

# Read the CSV file
with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile)

    # Skip the header row if present
    header = next(csv_reader)

    # Iterate through the rows
    for row in csv_reader:
        # Add the label from the 87th column (index 86) to the set
        if len(row) > 86:  # Ensure the row has at least 87 columns
            unique_labels.add(row[86])

# Convert the set to a sorted list
unique_labels_list = sorted(unique_labels)

# Print the unique labels
print("Unique Labels:", unique_labels_list)
length = len(unique_labels_list)

dictpredlist = []

for i in range(0,(length-1)):
    pred = label_encoder.inverse_transform([i])
    pred = str(pred[0])
    dictpredlist.append(pred)

print(dictpredlist)

#----------------------------------------------------------------------------------------------------------------


# Use MediaPipe Hands and Face Detection with specific parameters
with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands, \
     mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detection:

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1) # Flip the frame horizontally for a mirror effect
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert the frame to RGB as MediaPipe expects RGB input

        # Perform hand and face detection
        hand_results = hands.process(rgb_frame)
        face_results = face_detection.process(rgb_frame)

        # Initialize empty lists for left, right hand points, and head position
        left_list, right_list = [], []
        head_position = None

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
                #mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get coordinates for bounding box around the hand
                x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])
                y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0])
                x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])
                y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0])

                # Draw the bounding box around the hand
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Draw a black box at the top of the bounding box with "sign" label
                box_position = (x_min, y_min - 30)
                cv2.rectangle(frame, box_position, (x_min + 100, y_min), (0, 0, 0), -1)
                cv2.putText(frame, disp_label[2:-2], (x_min + 5, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Detect and display face bounding box if a face is detected
        if face_results.detections:
            for detection in face_results.detections:
                # Extract bounding box information for head position
                bbox = detection.location_data.relative_bounding_box
                x_min = int(bbox.xmin * frame.shape[1])
                y_min = int(bbox.ymin * frame.shape[0])
                width = int(bbox.width * frame.shape[1])
                height = int(bbox.height * frame.shape[0])
                x_max = x_min + width
                y_max = y_min + height

                # Approximate head position as the center of the face bounding box
                head_position = ((x_min + x_max) // 2, (y_min + y_max) // 2)

                # Draw the bounding box around the head
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                # Draw a black box at the top of the bounding box with "head" label
                box_position = (x_min, y_min - 30)
                cv2.rectangle(frame, box_position, (x_min + 50, y_min), (0, 0, 0), -1)
                cv2.putText(frame, "head", (x_min + 5, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        try:
            lmdisp = 'letter_mode:'+str(letter_mode)
            detected_sign_disp = disp_label[2:-2]
            cv2.rectangle(frame, (0,0), (300,150), (50, 0, 0), -1)
            cv2.putText(frame, lmdisp, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, detected_sign_disp, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        except:
            None

            # Increment the frame count

        frame_count += 1

        # Calculate the time difference between frames
        curr_time = time.time()
        time_diff = curr_time - prev_time

        # Calculate FPS if at least one second has passed
        if time_diff >= 1.0:
            fps = frame_count / time_diff
            prev_time = curr_time  # Update time for the next second
            frame_count = 0  # Reset frame count for the next second

        # Display FPS on the frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the output
        cv2.imshow("Hand and Head Detection", frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF

        if key == 115:
            if not detect_sentence:
                detect_sentence = True
            else:
                detect_sentence = False
        if key == ord('l'):
            if not letter_mode:
                letter_mode = True
            else:
                letter_mode = False
        if key == ord('i'):
            if not immediate_speak:
                immediate_speak = True
            else:
                immediate_speak = False

        if hand_results.multi_hand_landmarks:
            # Create Normalized Positions
            if head_position:
                normalized_head_x = head_position[0] / frame.shape[1]
                normalized_head_y = head_position[1] / frame.shape[0]

                if not left_list:
                    left_list = [(0, 0)] * 21
                if not right_list:
                    right_list = [(0, 0)] * 21
                major_list = left_list + right_list + [(normalized_head_x, normalized_head_y)]
                real_time_input = np.array(major_list).flatten().reshape(1, -1)  # Flatten and reshape for model input

                # Perform real-time prediction
                real_time_pred = model.predict(real_time_input)
                predicted_class = np.argmax(real_time_pred, axis=1)  # Multiclass classification
                disp_label = str(label_encoder.inverse_transform(predicted_class))
                print(str(predicted_class),str(label_encoder.inverse_transform(predicted_class)))

            else:
                print("Head Position: Not Detected")
                disp_label = "  No_symbol  "

            current_label = disp_label[2:-2]
            recent_labels.append(current_label)

        # Break the loop on 'Esc' key press
        if key == 27:  # ASCII for Esc key is 27
            break


# Release resources
cap.release()
cv2.destroyAllWindows()