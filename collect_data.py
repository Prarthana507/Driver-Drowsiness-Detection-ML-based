import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import csv
import os

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
HEAD_POSE_POINTS = [1, 33, 263, 61, 291, 199]

DOT_COLOR = (0, 255, 0)

# -----------------------------
# CSV Setup
# -----------------------------
csv_file = "driver_data.csv"

if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["EAR", "MAR", "Yaw", "Pitch", "Label"])

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

print("\n==============================")
print("PRESS KEYS TO SAVE DATA")
print("s = Safe")
print("d = Drowsy")
print("y = Yawning")
print("q = Quit")
print("==============================\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    h, w, _ = frame.shape

    ear, mar, yaw, pitch = 0, 0, 0, 0
    face_detected = False

    if results.multi_face_landmarks:
        face_detected = True

        for face in results.multi_face_landmarks:
            left_eye, right_eye, mouth = [], [], []
            face_2d, face_3d = [], []

            # Eyes
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

            # Mouth
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

            # Head Pose (optional)
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

            # Display values
            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    else:
        cv2.putText(frame, "No Face Detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Collect Driver Data", frame)

    key = cv2.waitKey(1)

    label = None
    if key == ord('s'):
        label = "Safe"
    elif key == ord('d'):
        label = "Drowsy"
    elif key == ord('y'):
        label = "Yawning"
    elif key == ord('q'):
        break

    # Save data
    if label is not None and face_detected:
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([ear, mar, yaw, pitch, label])

        print(f"Saved -> EAR:{ear:.2f}, MAR:{mar:.2f}, Label:{label}")

    elif label is not None and not face_detected:
        print("⚠ Face not detected. Data not saved.")

cap.release()
cv2.destroyAllWindows()