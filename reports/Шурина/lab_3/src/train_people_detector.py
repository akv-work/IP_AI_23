
from ultralytics import YOLO
import glob
import os

DATA_YAML = "data.yaml"
EPOCHS = 10
IMG_SIZE = 416
BATCH_SIZE = 4
MODEL_NAME = "people_detector"
CONF_THRESHOLD = 0.25
TEST_IMAGES_DIR = "../test/images"

print("Загрузка модели YOLOv11n...")
model = YOLO("yolo11n.pt")

print("Начало обучения...")
model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    name=MODEL_NAME
)
print("Обучение завершено!")

print("Валидация:")
metrics = model.val()
print("mAP:", metrics.mAP)

print("Визуализация...")
os.makedirs("predictions", exist_ok=True)

test_images = glob.glob(os.path.join(TEST_IMAGES_DIR, "*.jpg"))
for img_path in test_images:
    results = model.predict(img_path, conf=CONF_THRESHOLD, save=True)
    results.show()
    print(f"Обработка {img_path} завершена")

print("Готово! Посмотри папку 'predictions'.")
