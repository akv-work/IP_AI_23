
from ultralytics import YOLO
import os

MODEL_PATH = r"C:\Users\User\Desktop\Studing-7sem\OIIS\lab_3\yolo11n.pt"
VIDEO_PATH = "oaahh.mp4"  # путь к исходному видео
TRACKERS = ["botsort.yaml", "bytetrack.yaml"]  # трекеры для экспериментов
SAVE_DIR = "runs/track/"  # куда сохранять результаты

os.makedirs(SAVE_DIR, exist_ok=True)

print("Загрузка модели...")
model = YOLO(MODEL_PATH)

for tracker_file in TRACKERS:
    print(f"\nТрекинг с {tracker_file}...")

    results = model.track(
        source=VIDEO_PATH,
        tracker=tracker_file,
        persist=True,  # сохранять ID объектов
        show=True,  # показывать видео в реальном времени
        save=True  # сохранять видео
    )

    print(f"Готово! Видео с трекингом сохранено в {SAVE_DIR}")

print("\nВсе трекинги завершены!")
