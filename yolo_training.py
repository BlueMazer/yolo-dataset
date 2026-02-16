from ultralytics import YOLO

# Load YOLOv11 nano (deployment target = Jetson Orin Nano)
model = YOLO("yolo11n.pt")

model.train(
    data="data.yaml",
    epochs=200,          # small model → train longer
    imgsz=640,           # you can later deploy at 512 if needed
    batch=32,            # your 24GB GPU can handle this easily
    device=0,            # use CUDA GPU

    workers=8,           # matches your CPU cores
    cache=True,          # you have 50GB RAM — use it
    amp=True,            # mixed precision (faster + efficient)

    optimizer="AdamW",
    lr0=0.001,
    weight_decay=0.0005,

    mosaic=1.0,          # good for robustness
    mixup=0.1,
    degrees=8.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,

    patience=40,
    verbose=True
)
