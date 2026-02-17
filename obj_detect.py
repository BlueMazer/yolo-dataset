import cv2
from ultralytics import YOLO
import sys

# ============================
# CONFIG
# ============================
MODEL_PATH = "best_190_Epoch.pt"
Video = False  # True = video file, False = webcam
VIDEO_PATH = "/Users/rayanraad/Downloads/Video_20260214_161319_488.mp4"
MIN_CONF = 0.5

ARUCO_TYPE = "DICT_4X4_50"

# ============================
# LOAD MODEL
# ============================
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print("Failed to load YOLO model:", e)
    sys.exit()

# ============================
# OPEN SOURCE
# ============================
if Video:
    cap = cv2.VideoCapture(VIDEO_PATH)
else:
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # macOS safe backend

if not cap.isOpened():
    print("Error: Could not open source.")
    sys.exit()

# ============================
# VIDEO WRITER (ONLY FOR FILE)
# ============================
if Video:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # fallback

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

# ============================
# ARUCO SETUP
# ============================
aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_TYPE))
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# ============================
# MAIN LOOP
# ============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream ended or frame grab failed.")
        break

    # =======================
    # YOLO INFERENCE
    # =======================
    try:
        results = model(frame, verbose=False)

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < MIN_CONF:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame,
                            f"{label} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2)

    except Exception as e:
        print("YOLO error:", e)

    # =======================
    # ARUCO DETECTION
    # =======================
    try:
        corners, ids, _ = aruco_detector.detectMarkers(frame)

        if ids is not None:
            ids = ids.flatten()

            for marker_corners, marker_id in zip(corners, ids):

                pts = marker_corners[0].astype(int)

                # Draw marker border
                cv2.polylines(frame, [pts], True, (255, 0, 0), 2)

                # Compute center
                center_x = int(pts[:, 0].mean())
                center_y = int(pts[:, 1].mean())

                # Draw ID text
                cv2.putText(frame,
                            f"ID: {marker_id} | {ARUCO_TYPE}",
                            (center_x - 70, center_y - 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 0),
                            2)

    except Exception as e:
        print("ArUco error:", e)

    # =======================
    # DISPLAY
    # =======================
    cv2.imshow("YOLO + ArUco", frame)

    if Video:
        out.write(frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ============================
# CLEANUP
# ============================
cap.release()

if Video:
    out.release()

cv2.destroyAllWindows()
