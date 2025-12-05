
if __name__ == '__main__':
    from roboflow import Roboflow
    rf = Roboflow(api_key="PPvWh3zYDUwdXSmGOJai")
    project = rf.workspace("leo-ueno").project("people-detection-o4rdr")
    version = project.version(11)
    dataset = version.download("yolov11")

    from ultralytics import YOLO
    import os

    model = YOLO("yolo11s.pt")

    import torch
    device = 0 if torch.cuda.is_available() else "cpu"

    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=5,
        imgsz=640,
        batch=8,
        device=device,
    )

    results = model.predict(
        source="people.jpg",
        conf=0.25,
        save=True
    )


