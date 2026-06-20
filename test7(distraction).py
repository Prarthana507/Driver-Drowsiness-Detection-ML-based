import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import pygame
import math

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
def put_center_text(frame, text, y):
    h, w, _ = frame.shape
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
    x = (w - text_size[0]) // 2
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

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

# For Head Pose
HEAD_POSE_POINTS = [1, 33, 263, 61, 291, 199]

EAR_THRESHOLD = 0.23
MAR_THRESHOLD = 0.72

CLOSED_FRAMES = 15
YAWN_FRAMES = 22
DISTRACTION_FRAMES = 20

DOT_COLOR = (0, 255, 0)

# Counters
closed_count = 0
yawn_count = 0
left_count = 0
right_count = 0
down_count = 0

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

    drowsy = False
    yawning = False
    distracted = False
    distraction_text = ""

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:

            left_eye = []
            right_eye = []
            mouth = []

            face_2d = []
            face_3d = []

            # -----------------------------
            # Eye + Mouth Landmarks
            # -----------------------------
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

            for i in MOUTH:
                x = int(face.landmark[i].x * w)
                y = int(face.landmark[i].y * h)
                mouth.append((x, y))
                cv2.circle(frame, (x, y), 2, DOT_COLOR, -1)

            left_eye = np.array(left_eye)
            right_eye = np.array(right_eye)
            mouth = np.array(mouth)

            # -----------------------------
            # EAR + MAR
            # -----------------------------
            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2
            mar = calculate_mar(mouth)

            cv2.putText(frame, f"EAR:{ear:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"MAR:{mar:.2f}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # -----------------------------
            # Head Pose Points
            # -----------------------------
            for idx in HEAD_POSE_POINTS:
                lm = face.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)

                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = w
            cam_matrix = np.array([
                [focal_length, 0, w / 2],
                [0, focal_length, h / 2],
                [0, 0, 1]
            ])

            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                pitch = angles[0] * 360
                yaw = angles[1] * 360

                cv2.putText(frame, f"Yaw:{yaw:.1f}", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Pitch:{pitch:.1f}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # -----------------------------
                # Better Left / Right / Down
                # -----------------------------
                if yaw < -12:
                    left_count += 1
                    right_count = 0
                    down_count = 0

                elif yaw > 12:
                    right_count += 1
                    left_count = 0
                    down_count = 0

                elif pitch > 10:
                    down_count += 1
                    left_count = 0
                    right_count = 0

                else:
                    left_count = 0
                    right_count = 0
                    down_count = 0

            # -----------------------------
            # Yawning
            # -----------------------------
            if mar > MAR_THRESHOLD:
                yawn_count += 1
            else:
                yawn_count = 0

            if yawn_count > YAWN_FRAMES:
                yawning = True

            # -----------------------------
            # Distraction Trigger
            # -----------------------------
            if left_count > DISTRACTION_FRAMES:
                distracted = True
                distraction_text = "LOOKING LEFT"

            elif right_count > DISTRACTION_FRAMES:
                distracted = True
                distraction_text = "LOOKING RIGHT"

            elif down_count > DISTRACTION_FRAMES:
                distracted = True
                distraction_text = "LOOKING DOWN"

            # -----------------------------
            # Drowsiness
            # -----------------------------
            if ear < EAR_THRESHOLD and down_count < 8:
                closed_count += 1
            else:
                closed_count = 0

            if closed_count > CLOSED_FRAMES:
                drowsy = True

    # -----------------------------
    # Alerts
    # -----------------------------
    if drowsy:
        put_center_text(frame, "DROWSY!", h // 2)

    elif distracted:
        put_center_text(frame, f"DISTRACTED: {distraction_text}", h // 2)

    elif yawning:
        put_center_text(frame, "YAWNING!", h // 2)

    # -----------------------------
    # Alarm
    # -----------------------------
    if drowsy:
        if not alarm_playing:
            pygame.mixer.music.play(-1)
            alarm_playing = True
    else:
        if alarm_playing:
            pygame.mixer.music.stop()
            alarm_playing = False

    cv2.imshow("Smart Driver Safety System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
pygame.mixer.music.stop()
cv2.destroyAllWindows()