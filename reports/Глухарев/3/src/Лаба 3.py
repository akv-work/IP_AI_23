import os
import cv2
import requests
from PIL import Image
from ultralytics import YOLO
import yaml
import numpy as np
import torch

def main():
    """Основная функция выполнения лабораторной работы."""
    print("--- Проверка устройства ---")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device)
        print(f"GPU обнаружено: {gpu_name}")
        print(f"Всего GPU: {torch.cuda.device_count()}")
        print(f"Используемое устройство: {device} ({gpu_name})")
        training_device = 'cuda'
    else:
        print("GPU не обнаружено. Используется CPU.")
        training_device = 'cpu'

    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = PROJECT_DIR
    DATA_CONFIG_PATH = os.path.join(DATASET_PATH, "data.yaml")

    MODEL_NAME = "yolov10m.pt"

    SAVE_DIR = os.path.join(PROJECT_DIR, "runs", "detect", "water_meter_train")
    os.makedirs(SAVE_DIR, exist_ok=True)

    OUTPUT_DIR = os.path.join(PROJECT_DIR, "detection_results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n--- Этап 1: Анализ датасета ---")
    print(f"Датасет находится в: {DATASET_PATH}")
    print(f"Конфигурация датасета: {DATA_CONFIG_PATH}")

    with open(DATA_CONFIG_PATH, 'r') as f:
        data_config = yaml.safe_load(f)

    print(f"Количество классов: {data_config.get('nc', 'Не указано')}")
    print(f"Названия классов: {data_config.get('names', 'Не указаны')}")
    print(f"Путь к тренировочным изображениям (из config): {data_config.get('train', 'Не указан')}")
    print(f"Путь к валидационным изображениям (из config): {data_config.get('val', 'Не указан')}")
    print(f"Путь к тестовым изображениям (из config): {data_config.get('test', 'Не указан')}")

    print("\n--- Этап 2: Обучение модели YOLOv10m ---")
    print("Загрузка модели...")
    model = YOLO(MODEL_NAME)
    print(f"Модель будет использовать устройство: {next(model.model.parameters()).device}")
    print(f"Запуск обучения на устройстве: {training_device}")

    results = model.train(
        data=DATA_CONFIG_PATH,
        epochs=25, # Изменено на 25 эпох
        imgsz=608,
        batch=16,
        save_dir=SAVE_DIR,
        name="water_meter_experiment",
        exist_ok=True,
        device=training_device,
        workers=0 # <--- Вот этот параметр важен
    )

    print("\n--- Этап 3: Оценка эффективности на валидационной выборке ---")
    # Укажите workers=0 и для валидации, если она использует DataLoader
    validation_results = model.val(data=DATA_CONFIG_PATH, split='val', device=training_device, workers=0)
    print(f"Результаты валидации: {validation_results}")
    print(f"mAP@0.5: {validation_results.box.map50:.4f}")
    print(f"mAP@0.5:0.95 (среднее по IoU): {validation_results.box.map:.4f}")

    print("\n--- Этап 4: Визуализация работы детектора на изображениях из датасета ---")


    val_images_path = os.path.join(DATASET_PATH, data_config.get('val', 'valid/images')) # Предполагаем формат из Roboflow
    if not os.path.exists(val_images_path):
        print(f"Папка с валидационными изображениями не найдена: {val_images_path}")
        print("Попробуем использовать путь из data.yaml напрямую (возможно, он относительный).")
        val_images_path = os.path.join(os.path.dirname(DATA_CONFIG_PATH), data_config.get('val', 'valid/images'))
        if not os.path.exists(val_images_path):
             print(f"Папка с валидационными изображениями по пути из data.yaml тоже не найдена: {val_images_path}")
             return # Выходим, если не можем найти изображения
        else:
             print(f"Путь к валидационным изображениям найден: {val_images_path}")

    image_files = [f for f in os.listdir(val_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not image_files:
        print(f"В папке {val_images_path} не найдено изображений.")
        return

    num_images_to_visualize = min(5, len(image_files)) # Визуализируем первые 5 изображений или все, если их меньше
    print(f"Будет визуализировано {num_images_to_visualize} изображений из валидационной выборки.")

    for i in range(num_images_to_visualize):
        image_filename = image_files[i]
        image_path = os.path.join(val_images_path, image_filename)

        try:
            print(f"Обработка изображения {i+1}: {image_filename}")
            image = cv2.imread(image_path)

            if image is None:
                print(f"Ошибка: не удалось загрузить изображение {image_path}")
                continue

            results = model(image)

            annotated_frame = results[0].plot() # plot() рисует боксы и метки

            output_filename = os.path.join(OUTPUT_DIR, f"detected_val_image_{i+1}_{image_filename}")
            cv2.imwrite(output_filename, annotated_frame)
            print(f"Результат сохранен как: {output_filename}")

        except Exception as e:
            print(f"Ошибка при обработке изображения {image_path}: {e}")

    print("\n--- Лабораторная работа завершена ---")
    print("1. Датасет проанализирован.")
    print("2. Модель YOLOv10m обучена.")
    print("3. Эффективность оценена (mAP).")
    print("4. Результаты обнаружения на изображениях из датасета сохранены в папку 'detection_results'.")
    print("Теперь вы можете оформить отчет и загрузить код и отчет в репозиторий.")


if __name__ == '__main__':
    main()