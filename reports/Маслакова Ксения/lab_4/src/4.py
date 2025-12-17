import cv2
from ultralytics import YOLO

def run_tracking(model_path, video_path, tracker_config_file, output_name):
    """
    Функция для запуска трекинга с определенным алгоритмом.
    """
    print(f"--- Запуск трекинга с конфигом: {tracker_config_file} ---")
    
    # 1. Загрузка модели (Используем Нашу обученную модель из ЛР 3)
    model = YOLO(model_path)

    # 2. Запуск трекинга
    # persist=True важен для сохранения ID объектов между кадрами
    results = model.track(
        source=video_path, 
        tracker=tracker_config_file, 
        conf=0.5,      # Порог уверенности (можно менять для пункта 3)
        iou=0.5,       # Порог IOU (можно менять для пункта 3)
        show=True,     # Показывать видео в реальном времени
        save=True,     # Сохранить результат
        project='runs/track', # Папка для сохранения
        name=output_name,     # Имя подпапки
        persist=True          # Важно для трекинга!
    )
    
    print(f"Готово. Результаты сохранены в runs/track/{output_name}")

if __name__ == "__main__":
    # ПУТЬ К НАШЕЙ МОДЕЛИ ИЗ ЛР 3
    MY_MODEL = r"D:\4 Курс\Oiiis\runs\detect\cat_yolov10_result_win\weights\best.pt" 
    
    # ПУТЬ К ВИДЕО
    MY_VIDEO = r"C:\Users\KSENIA\Downloads\cat.mp4" 

    # Эксперимент 1: Использование BoT-SORT
    # botsort.yaml - это встроенный конфиг в ultralytics
    run_tracking(MY_MODEL, MY_VIDEO, "botsort.yaml", "experiment_botsort")

    # Эксперимент 2: Использование ByteTrack 
    # bytetrack.yaml - это встроенный конфиг в ultralytics
    run_tracking(MY_MODEL, MY_VIDEO, "bytetrack.yaml", "experiment_bytetrack")