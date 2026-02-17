import cv2
from ultralytics import YOLO
import sys

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "runs\detect\train7\weights\best.pt"   # change to last.pt if needed
SOURCE = 0               # 0 = webcam, or put image path, video path, or folder

# ----------------------------
# Load YOLO model
# ----------------------------
model = YOLO(MODEL_PATH)

# ----------------------------
# Open video/image source
# ----------------------------
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open source.")
    sys.exit()

# ----------------------------
# Optional: ArUco setup
# ----------------------------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# ----------------------------
# Main loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ------------------------
    # YOLO inference
    # ------------------------
    results = model(frame, verbose=False)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,0),
                        2)

    # ------------------------
    # ArUco detection
    # ------------------------
    corners, ids, _ = aruco_detector.detectMarkers(frame)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    # ------------------------
    # Show output
    # ------------------------
    cv2.imshow("YOLO + ArUco", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
