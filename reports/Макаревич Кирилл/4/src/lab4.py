import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path


MODEL_WEIGHTS = r"D:\Kirill\OIIS\lab3\weights\best.pt"

VIDEOS_DIR = r"D:\Kirill\OIIS\lab4\video"

OUTPUT_DIR = "tracking_results"
 
TRACKER_CONFIGS = [
    "bytetrack.yaml",  
    "botsort.yaml",    
]

EXPERIMENT_PARAMS = [
    {"name": "low_conf", "conf": 0.15, "iou": 0.45, "imgsz": 640},
    {"name": "high_conf", "conf": 0.50, "iou": 0.45, "imgsz": 640},
]


def list_videos(video_dir, exts=(".mp4", ".avi", ".mov", ".mkv")):
    p = Path(video_dir)
    if not p.exists():
        print(f"[ОШИБКА] Папка с видео не найдена: {video_dir}")
        return []

    videos = [str(f) for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts]
    if not videos:
        print(f"[ПРЕДУПРЕЖДЕНИЕ] В папке {video_dir} нет видео с расширениями {exts}")
    return videos


def visualize_image(img_path, title="Result"):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


def run_default_tracking(video_paths):
    for tracker in TRACKER_CONFIGS:
        print("=" * 80)
        print(f"Standard run | Tracker: {tracker}")
        print("=" * 80)

        model = YOLO(MODEL_WEIGHTS)

        for video_path in video_paths:
            video_name = Path(video_path).stem
            run_name = f"default_{Path(tracker).stem}_{video_name}"

            print(f"\n[Трекинг] Видео: {video_path}")
            print(f"[Трекинг] Трекер: {tracker}")

            model.track(
                source=video_path,
                tracker=tracker,
                save=True,
                project=OUTPUT_DIR,
                name=run_name,
                exist_ok=True,
            )


def run_parameter_experiments(video_paths):
    model = YOLO(MODEL_WEIGHTS)

    for exp in EXPERIMENT_PARAMS:
        for tracker in TRACKER_CONFIGS:
            print("=" * 80)
            print(f"Custom run: {exp['name']} | Tracker: {tracker}")
            print(f"conf={exp['conf']}  iou={exp['iou']}  imgsz={exp['imgsz']}")
            print("=" * 80)

            for video_path in video_paths:
                video_name = Path(video_path).stem
                run_name = f"exp_{exp['name']}_{Path(tracker).stem}_{video_name}"

                print(f"\n[Эксперимент] Видео: {video_path}")

                model.track(
                    source=video_path,
                    tracker=tracker,
                    conf=exp["conf"],
                    iou=exp["iou"],
                    imgsz=exp["imgsz"],
                    save=True,
                    project=OUTPUT_DIR,
                    name=run_name,
                    exist_ok=True,
                )


def main():
    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(
            f"Не найден файл модели (best.pt). Проверьте путь MODEL_WEIGHTS: {MODEL_WEIGHTS}"
        )

    video_paths = list_videos(VIDEOS_DIR)
    if not video_paths:
        raise FileNotFoundError(
            f"В папке с видео ({VIDEOS_DIR}) нет подходящих файлов. "
            f"Скачайте несколько видео с множественными объектами классов из lab3 "
            f"и положите их туда."
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    run_default_tracking(video_paths)

    run_parameter_experiments(video_paths)


if __name__ == "__main__":
    main()
