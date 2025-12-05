import os
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

# --- Загрузка датасета из Roboflow ---
rf = Roboflow(api_key="u1g4JSOm4Hf8ZtQAYkwU")  # замените на ваш ключ
project = rf.workspace("leo-ueno").project("people-detection-o4rdr")
version = project.version(10)  # нужная версия датасета
dataset = version.download("yolov11")  # формат YOLOv11

print("Датасет загружен. Путь:", dataset.location)

# --- Анализ датасета ---
def analyze_dataset(path):
    data_yaml = os.path.join(path, "data.yaml")
    with open(data_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
    print("Классы:", cfg.get('names'), " nc:", cfg.get('nc'))

    for split in ['train', 'valid', 'test']:
        images_dir = os.path.join(path, split, "images")
        if os.path.exists(images_dir):
            images = [f for f in os.listdir(images_dir)
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            print(split, ":", len(images), "изображений")
        else:
            print(f"{split}/images не найдено!")

analyze_dataset(dataset.location)

# --- Загрузка модели YOLOv11 ---
model = YOLO("yolo11n.pt")  # можно yolo11s.pt, yolo11m.pt и др.

# --- Функция для получения первого изображения из сплита ---
def get_first_image(split):
    images_dir = os.path.join(dataset.location, split, "images")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Папка {split}/images не найдена!")
    img_path = next((os.path.join(images_dir, f) for f in os.listdir(images_dir)
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))), None)
    if img_path is None:
        raise FileNotFoundError(f"В {split}/images нет изображений!")
    return img_path

# --- Инференс на изображении ---
def run_inference(img_path=None):
    if img_path is None:
        img_path = get_first_image("train")
    print("Используем изображение для инференса:", img_path)

    results_list = model(img_path)  # inference (возвращает список)
    result = results_list[0]        # берем первый элемент
    result.show()   # отображаем bbox
    result.save()   # сохраняем в runs/detect/predict/
    result.print()  # вывод классов и confidence

# Инференс на первом изображении train
run_inference()

# --- Визуализация примеров из train, valid, test ---
for split in ["train", "valid", "test"]:
    try:
        img_path = get_first_image(split)
        res = model(img_path)[0]
        res.show()
        print(f"Пример из {split}: {os.path.basename(img_path)}")
    except FileNotFoundError:
        print(f"Пропущен сплит {split}, нет изображений.")
