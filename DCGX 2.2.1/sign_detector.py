import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import mediapipe as mp
import customtkinter as ctk
import threading
import time
import json
import sys

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # PyInstaller unpack folder
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class SignDetectionFrame(ctk.CTkFrame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)

        # Models and labels setup
        self.mp_hands = mp.solutions.hands
        self.mp_face_detection = mp.solutions.face_detection

        number_model_name = "DCGX-Numbers"
        self.model_numerical = load_model(resource_path(f"appdata/models/{number_model_name}.keras"))
        numerical_label_list_path = resource_path(f"appdata/models/{number_model_name}.keras_labels.json")

        number_model_name = "DCGX-ISL-Alphabet"
        self.model_alphabet = load_model(resource_path(f"appdata/models/{number_model_name}.keras"))
        alphabet_label_list_path = resource_path(f"appdata/models/{number_model_name}.keras_labels.json")

        main_model_name = "DCGX-Main"
        self.model_main = load_model(resource_path(f"appdata/models/{main_model_name}.keras"))
        main_label_list_path = resource_path(f"appdata/models/{main_model_name}.keras_labels.json")

        self.number_label_encoder = LabelEncoder()
        self.number_label_encoder.fit(self.load_labels(numerical_label_list_path))

        self.main_label_encoder = LabelEncoder()
        self.main_label_encoder.fit(self.load_labels(main_label_list_path))

        self.alphabet_label_encoder = LabelEncoder()
        self.alphabet_label_encoder.fit(self.load_labels(alphabet_label_list_path))

        # UI Elements
        self.model_choice_var = ctk.StringVar(value="Main")
        self.dropdown = ctk.CTkOptionMenu(self, values=["Main", "Numbers","Alphabet"], variable=self.model_choice_var)
        self.dropdown.pack(pady=15)

        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.pack(pady=10)

        # MediaPipe init
        self.cap = cv2.VideoCapture(0)
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                         min_detection_confidence=0.6, min_tracking_confidence=0.5)
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.7)

        # FPS and display vars
        self.frame_count = 0
        self.prev_time = time.time()
        self.fps = 0
        self.disp_label = ''
        self.confidence = ''
        self.recent_labels = []

        # Thread control flags
        self.running = False
        self.thread = None

    def load_labels(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def start_detection(self):
        if self.running:
            return  # Already running, chill
        self.running = True
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.update_loop, daemon=True)
            self.thread.start()

    def stop_detection(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1)
            self.thread = None
        if self.cap.isOpened():
            self.cap.release()

    def update_loop(self):
        while self.running:
            self.update_frame()
            time.sleep(0.01)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = self.hands.process(rgb_frame)
        face_results = self.face_detection.process(rgb_frame)

        left_list, right_list = [], []
        head_position = None

        # Hand detection + bounding boxes + label display
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                hand_label = hand_results.multi_handedness[idx].classification[0].label
                hand_points = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

                if hand_label == 'Left':
                    left_list = hand_points
                else:
                    right_list = hand_points

                x_min = int(min(lm.x for lm in hand_landmarks.landmark) * frame.shape[1])
                y_min = int(min(lm.y for lm in hand_landmarks.landmark) * frame.shape[0])
                x_max = int(max(lm.x for lm in hand_landmarks.landmark) * frame.shape[1])
                y_max = int(max(lm.y for lm in hand_landmarks.landmark) * frame.shape[0])

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.rectangle(frame, (x_min, y_min - 30), (x_min + 100, y_min), (0, 0, 0), -1)
                cv2.putText(frame, self.disp_label[2:-2], (x_min + 5, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Face detection + bounding box
        if face_results.detections:
            for detection in face_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x_min = int(bbox.xmin * frame.shape[1])
                y_min = int(bbox.ymin * frame.shape[0])
                x_max = x_min + int(bbox.width * frame.shape[1])
                y_max = y_min + int(bbox.height * frame.shape[0])
                head_position = ((x_min + x_max) // 2, (y_min + y_max) // 2)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.rectangle(frame, (x_min, y_min - 30), (x_min + 50, y_min), (0, 0, 0), -1)
                cv2.putText(frame, "head", (x_min + 5, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Prediction logic
        if hand_results.multi_hand_landmarks:
            if head_position:
                norm_head_x = head_position[0] / frame.shape[1]
                norm_head_y = head_position[1] / frame.shape[0]

                if not left_list:
                    left_list = [(0, 0)] * 21
                if not right_list:
                    right_list = [(0, 0)] * 21

                major_list = left_list + right_list + [(norm_head_x, norm_head_y)]
                real_time_input = np.array(major_list).flatten().reshape(1, -1)

                current_model = self.model_choice_var.get().lower()
                if current_model == "numbers":
                    real_time_pred = self.model_numerical.predict(real_time_input)
                    predicted_class = np.argmax(real_time_pred, axis=1)
                    self.disp_label = str(self.number_label_encoder.inverse_transform(predicted_class))
                elif current_model == "main":
                    real_time_pred = self.model_main.predict(real_time_input)
                    predicted_class = np.argmax(real_time_pred, axis=1)
                    self.disp_label = str(self.main_label_encoder.inverse_transform(predicted_class))
                else:
                    real_time_pred = self.model_alphabet.predict(real_time_input)
                    predicted_class = np.argmax(real_time_pred, axis=1)
                    self.disp_label = str(self.alphabet_label_encoder.inverse_transform(predicted_class))

                self.confidence = str(int(np.max(real_time_pred, axis=1)[0] * 100))
            else:
                self.disp_label = "  No_symbol  "

            self.recent_labels.append(self.disp_label[2:-2])

        # Info overlay
        try:
            label_text = "Sign:" + self.disp_label[2:-2] + " " + self.confidence + "%"
        except:
            label_text = "Sign:"
        length = max((len(label_text)*25), 200)
        cv2.rectangle(frame, (0, 0), (length, 100), (50, 0, 0), -1)

        if face_results.detections:
            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                cv2.putText(frame, label_text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Sign: None", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "No Person", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # FPS calc
        self.frame_count += 1
        curr_time = time.time()
        if curr_time - self.prev_time >= 1.0:
            self.fps = self.frame_count / (curr_time - self.prev_time)
            self.prev_time = curr_time
            self.frame_count = 0

        cv2.putText(frame, f"FPS: {self.fps:.2f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display frame on CTkLabel
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_ctk = ctk.CTkImage(light_image=img_pil, size=(640, 480))
        self.video_label.configure(image=img_ctk)
        self.video_label.image = img_ctk

    def destroy(self):
        self.stop_detection()
        super().destroy()


def run_app():
    app = ctk.CTk()
    app.geometry("700x600")
    app.title("DCGX-2.1.0")
    app.iconbitmap(resource_path('l0g0_dcg.ico'))

    frame = SignDetectionFrame(app)
    frame.pack(fill="both", expand=True)

    app.mainloop()

if __name__ == "__main__":
    run_app()
