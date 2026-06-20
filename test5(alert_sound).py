import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import pygame
pygame.mixer.init()
pygame.mixer.music.load("sounds/alarm.wav")
alarm_playing = False
def calculate_ear(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EAR_THRESHOLD = 0.23
CLOSED_FRAMES_THRESHOLD = 15
closed_frame_count = 0
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Could not open webcam")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    h, w, _ = frame.shape
    drowsy_detected = False
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_points = []
            right_eye_points = []
            for idx in LEFT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                left_eye_points.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            for idx in RIGHT_EYE:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                right_eye_points.append((x, y))
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            left_eye_points = np.array(left_eye_points)
            right_eye_points = np.array(right_eye_points)
            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)
            avg_ear = (left_ear + right_ear) / 2.0
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if avg_ear < EAR_THRESHOLD:
                closed_frame_count += 1
            else:
                closed_frame_count = 0
            if closed_frame_count >= CLOSED_FRAMES_THRESHOLD:
                drowsy_detected = True
                cv2.putText(frame, "DROWSINESS ALERT!", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    if drowsy_detected:
        if not alarm_playing:
            pygame.mixer.music.play(-1)
            alarm_playing = True
    else:
        if alarm_playing:
            pygame.mixer.music.stop()
            alarm_playing = False
    cv2.imshow("Drowsiness Detection with Alarm", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
pygame.mixer.music.stop()
cv2.destroyAllWindows()