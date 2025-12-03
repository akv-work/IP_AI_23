from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")

model.track(
    source="video.mp4",
    conf=0.25,
    iou=0.45,
    tracker="botsort.yaml",
    show=False,
    save=True,
    name="botsort"
)

model.track(
    source="video.mp4",
    conf=0.25,
    iou=0.45,
    tracker="bytetrack.yaml",
    show=False,
    save=True,
    name="bytetrack"
)
