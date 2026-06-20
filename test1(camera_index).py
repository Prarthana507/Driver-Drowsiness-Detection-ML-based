import cv2
camera_index = 1   
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Could not open camera")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break
    cv2.imshow(f"Camera {camera_index}", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()