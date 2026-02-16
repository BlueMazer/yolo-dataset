from ultralytics import YOLO

# Load YOLOv11 nano model (fast for CPU)
model = YOLO("yolo11n.pt")

# Train
model.train(
    data="data.yaml",  # make sure this path points to your dataset.yaml
    epochs=100,           # adjust if you want fewer for testing
    imgsz=640,            # input image size
    batch=4,              # CPU-friendly batch size
    device="cpu"          # force CPU
)