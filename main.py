import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import pygame
import joblib

# -----------------------------
# Load ML Model
# -----------------------------
model = joblib.load("driver_state_model.pkl")

# -----------------------------
# Alarm setup
# -----------------------------
pygame.mixer.init()
pygame.mixer.music.load("sounds/alarm.wav")
alarm_playing = False

# -----------------------------
# EAR Calculation
# -----------------------------
def calculate_ear(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

# -----------------------------
# MAR Calculation
# -----------------------------
def calculate_mar(mouth_points):
    A = dist.euclidean(mouth_points[1], mouth_points[7])
    B = dist.euclidean(mouth_points[2], mouth_points[6])
    C = dist.euclidean(mouth_points[3], mouth_points[5])
    D = dist.euclidean(mouth_points[0], mouth_points[4])
    return (A + B + C) / (2.0 * D)

# -----------------------------
# Center Text
# -----------------------------
def put_center_text(frame, text, y, color=(0, 0, 255), scale=1.0, thickness=3):
    h, w, _ = frame.shape
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    x = (w - text_size[0]) // 2
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

# -----------------------------
# MediaPipe Setup
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# -----------------------------
# Landmark Indexes
# -----------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 82, 13, 312, 308, 317, 14, 87]

# -----------------------------
# Thresholds
# -----------------------------
EAR_THRESHOLD = 0.23
CLOSED_FRAMES = 15

DOT_COLOR = (0, 255, 0)
closed_count = 0

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    h, w, _ = frame.shape
    ml_state = "SAFE"
    display_state = "SAFE"
    drowsy_alert = False

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:

            left_eye, right_eye, mouth = [], [], []

            # Eye landmarks
            for i in LEFT_EYE:
                x = int(face.landmark[i].x * w)
                y = int(face.landmark[i].y * h)
                left_eye.append((x, y))
                cv2.circle(frame, (x, y), 2, DOT_COLOR, -1)

            for i in RIGHT_EYE:
                x = int(face.landmark[i].x * w)
                y = int(face.landmark[i].y * h)
                right_eye.append((x, y))
                cv2.circle(frame, (x, y), 2, DOT_COLOR, -1)

            # Mouth landmarks
            for i in MOUTH:
                x = int(face.landmark[i].x * w)
                y = int(face.landmark[i].y * h)
                mouth.append((x, y))
                cv2.circle(frame, (x, y), 2, DOT_COLOR, -1)

            left_eye = np.array(left_eye)
            right_eye = np.array(right_eye)
            mouth = np.array(mouth)

            # EAR & MAR
            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2
            mar = calculate_mar(mouth)

            # ML Prediction (Yawning only)
            features = np.array([[ear, mar, 0, 0]])
            prediction = model.predict(features)[0]
            ml_state = prediction.upper()

            # -----------------------------
            # Drowsiness Logic
            # -----------------------------
            if ear < EAR_THRESHOLD:
                closed_count += 1
            else:
                closed_count = 0

            if closed_count > CLOSED_FRAMES:
                drowsy_alert = True

            # -----------------------------
            # FINAL DISPLAY STATE (NO FLICKER)
            # -----------------------------
            if drowsy_alert:
                display_state = "DROWSY"
            elif ml_state == "YAWNING":
                display_state = "YAWNING"
            else:
                display_state = "SAFE"

            # Show state
            cv2.putText(frame, f"STATE: {display_state}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # -----------------------------
    # ALERTS
    # -----------------------------
    if drowsy_alert:
        put_center_text(frame, "DROWSINESS ALERT!", h // 2)

    elif display_state == "YAWNING":
        put_center_text(frame, "YAWNING DETECTED!", h // 2)

    # -----------------------------
    # Alarm
    # -----------------------------
    if drowsy_alert:
        if not alarm_playing:
            pygame.mixer.music.play(-1)
            alarm_playing = True
    else:
        if alarm_playing:
            pygame.mixer.music.stop()
            alarm_playing = False

    cv2.imshow("Driver Monitoring System (Final)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
pygame.mixer.music.stop()
cv2.destroyAllWindows()