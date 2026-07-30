from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=5,
    imgsz=320,
    batch=4,
    workers=2,
    name="pothole_detector"
)