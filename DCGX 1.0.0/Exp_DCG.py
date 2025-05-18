import cv2
import mediapipe as mp
import pandas as pd
import os

# Initialize MediaPipe hands, face detection, and drawing utilities
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Start the camera feed
cap = cv2.VideoCapture(0)

# Create headers for 86 columns (point_1 to point_86) and the last column 'label'
headers = [f"point_{i}" for i in range(1, 87)] + ["label"]
all_data = []
count = 0

label = str(input("Label:"))

# Use MediaPipe Hands and Face Detection with specific parameters
with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands, \
     mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detection:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        # Convert the frame to RGB as MediaPipe expects RGB input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get coordinates for bounding box around the hand
                x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])
                y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0])
                x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * frame.shape[1])
                y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * frame.shape[0])

                # Draw the bounding box around the hand
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Draw a black box at the top of the bounding box with "sign" label
                box_position = (x_min, y_min - 30)
                cv2.rectangle(frame, box_position, (x_min + 50, y_min), (0, 0, 0), -1)
                cv2.putText(frame, "sign", (x_min + 5, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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

        # Display the output
        cv2.imshow("Hand and Head Detection", frame)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF

        # If 'p' is pressed, print the normalized coordinates for left and right hands
        if key == ord('p'):
            print("Left Hand Normalized Coordinates:", left_list)
            print("Right Hand Normalized Coordinates:", right_list)

            # Print normalized head position
            if head_position:
                normalized_head_x = head_position[0] / frame.shape[1]
                normalized_head_y = head_position[1] / frame.shape[0]
                head_list = [(normalized_head_x, normalized_head_y),label]
                print("Head Normalized Position:", head_list)
                if not left_list:
                    left_list = [(0, 0)] * 21
                if not right_list:
                    right_list = [(0, 0)] * 21
                major_list = left_list + right_list + head_list
                print("major_list:",major_list)

                # Separate the label and flatten the rest of the data
                flattened_data = [value for point in major_list[:-1] for value in point]
                flattened_data.append(major_list[-1])  # Append the label at the end of the row

                all_data.append(flattened_data)
                count +=1

            else:
                print("Head Position: Not Detected")

        # Break the loop on 'Esc' key press
        if key == 27:  # ASCII for Esc key is 27
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

file_name = 'points_web_demo.csv'

# Check if points.csv already exists
file_exists = os.path.isfile(file_name)

# Convert collected data to a DataFrame and append it to points.csv
df = pd.DataFrame(all_data, columns=headers)
df.to_csv(file_name, mode='a', header=not file_exists, index=False)

print("Saved", str(count), "points to" ,file_name,". Exiting.")

