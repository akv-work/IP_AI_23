import os
import sys
import shutil
import numpy as np
import pandas as pd
import yaml
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
import glob
from datetime import datetime
import json
import gc


class CSVToYOLOConverter:
    def __init__(self, base_path):
        self.base_path = base_path
        self.class_name = "person"
        self.class_id = 0

    def convert_annotations(self, csv_path, images_dir, output_dir):
        """Конвертация CSV аннотаций в YOLO формат"""
        labels_dir = os.path.join(output_dir, 'labels')
        images_output_dir = os.path.join(output_dir, 'images')

        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(images_output_dir, exist_ok=True)

        print(f"Чтение CSV: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
            print(f"Загружено {len(df)} аннотаций из {csv_path}")
        except Exception as e:
            print(f"Ошибка чтения CSV: {e}")
            return 0

        processed_files = 0
        missing_images = 0

        for filename, group in df.groupby('filename'):
            src_image_path = os.path.join(images_dir, filename)

            if not os.path.exists(src_image_path):
                missing_images += 1
                continue

            dst_image_path = os.path.join(images_output_dir, filename)
            shutil.copy2(src_image_path, dst_image_path)

            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_filename)

            try:
                with open(label_path, 'w') as f:
                    for _, row in group.iterrows():
                        x_center, y_center, width, height = self.convert_to_yolo_format(
                            row['xmin'], row['ymin'], row['xmax'], row['ymax'],
                            row['width'], row['height']
                        )
                        f.write(f"{self.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                processed_files += 1
            except Exception as e:
                print(f"Ошибка создания аннотации для {filename}: {e}")

        if missing_images > 0:
            print(f"Всего отсутствующих изображений: {missing_images}")

        print(f"Обработано {processed_files} файлов для {output_dir}")
        return processed_files

    def convert_to_yolo_format(self, xmin, ymin, xmax, ymax, img_width, img_height):
        """Конвертация координат в YOLO формат"""
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))

        return x_center, y_center, width, height


class PeopleDetectorTrainer:
    def __init__(self, dataset_path='lab3'):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = os.path.join(self.script_dir, dataset_path)
        self.yolo_dataset_path = os.path.join(self.script_dir, 'yolo_dataset')
        self.model = None
        self.results = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.script_dir, 'training_results', self.timestamp)

    def setup_environment(self):
        """Проверка и настройка окружения"""
        print("Настройка окружения...")

        torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"GPU доступен: {gpu_name} ({gpu_memory:.1f} GB)")
            self.device = 0
            # Очистка памяти
            torch.cuda.empty_cache()
            # Установка лимита памяти
            torch.cuda.set_per_process_memory_fraction(0.7)  # Используем только 70% памяти
        else:
            print("GPU не доступен, используется CPU")
            self.device = None

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'detections'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'graphs'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'models'), exist_ok=True)

    def prepare_dataset(self):
        """Подготовка датасета в YOLO формате"""
        print("Подготовка датасета в YOLO формате...")

        if not os.path.exists(self.dataset_path):
            print(f"Директория датасета не найдена: {self.dataset_path}")
            return False

        converter = CSVToYOLOConverter(self.dataset_path)

        yolo_dirs = ['train', 'val', 'test']
        for dir_name in yolo_dirs:
            os.makedirs(os.path.join(self.yolo_dataset_path, dir_name, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.yolo_dataset_path, dir_name, 'labels'), exist_ok=True)

        train_csv = os.path.join(self.dataset_path, 'train', '_annotations.csv')
        train_images = os.path.join(self.dataset_path, 'train')
        print("Конвертация тренировочных данных...")
        converter.convert_annotations(train_csv, train_images,
                                      os.path.join(self.yolo_dataset_path, 'train'))

        val_csv = os.path.join(self.dataset_path, 'valid', '_annotations.csv')
        val_images = os.path.join(self.dataset_path, 'valid')
        print("Конвертация валидационных данных...")
        converter.convert_annotations(val_csv, val_images,
                                      os.path.join(self.yolo_dataset_path, 'val'))

        test_csv = os.path.join(self.dataset_path, 'test', '_annotations.csv')
        test_images = os.path.join(self.dataset_path, 'test')
        print("Конвертация тестовых данных...")
        converter.convert_annotations(test_csv, test_images,
                                      os.path.join(self.yolo_dataset_path, 'test'))

        self.create_data_yaml()
        print("Датасет подготовлен в YOLO формате")
        return True

    def create_data_yaml(self):
        """Создание data.yaml файла"""
        data = {
            'path': os.path.abspath(self.yolo_dataset_path),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 1,
            'names': ['person']
        }

        yaml_path = os.path.join(self.yolo_dataset_path, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        print(f"data.yaml создан: {yaml_path}")

    def analyze_dataset(self):
        """Анализ датасета"""
        print("Анализ датасета:")

        splits = ['train', 'val', 'test']
        for split in splits:
            images_dir = os.path.join(self.yolo_dataset_path, split, 'images')
            labels_dir = os.path.join(self.yolo_dataset_path, split, 'labels')

            if not os.path.exists(images_dir):
                print(f"Директория {images_dir} не найдена")
                continue

            num_images = len([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
            num_labels = len([f for f in os.listdir(labels_dir) if f.endswith('.txt')]) if os.path.exists(
                labels_dir) else 0

            total_objects = 0
            if os.path.exists(labels_dir):
                for label_file in os.listdir(labels_dir):
                    if label_file.endswith('.txt'):
                        try:
                            with open(os.path.join(labels_dir, label_file), 'r') as f:
                                total_objects += len(f.readlines())
                        except:
                            pass

            print(f"  {split.capitalize()}: {num_images} изображений, {num_labels} аннотаций, {total_objects} объектов")

    def setup_model(self):
        """Инициализация модели YOLOv11n"""
        print("Инициализация модели YOLOv11n...")

        try:
            self.model = YOLO('yolo11n.pt')
            print("Модель успешно загружена")
            return True
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False

    def train_model(self, epochs=50, imgsz=640, batch=12, patience=15):
        """
        Обучение модели с безопасными параметрами
        """
        print("Запуск обучения модели...")

        if self.model is None:
            print("Модель не инициализирована")
            return False

        try:
            # Безопасные параметры для избежания "Killed"
            self.results = self.model.train(
                data=os.path.join(self.yolo_dataset_path, 'data.yaml'),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,  # Уменьшили batch size
                device=self.device,
                workers=4,  # Уменьшили workers
                lr0=0.01,
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3.0,
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                box=7.5,
                cls=0.5,
                dfl=1.5,
                patience=patience,
                save=True,
                save_period=10,
                cache=False,  # Отключили кэш для экономии памяти
                name=f'people_detection_{self.timestamp}',
                exist_ok=True,
                amp=True,  # Mixed precision обязательно
                # Минимальные аугментации
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=0.0,
                translate=0.1,
                scale=0.5,
                shear=0.0,
                perspective=0.0,
                flipud=0.0,
                fliplr=0.5,
                mosaic=0.0,  # Отключили mosaic для экономии памяти
                mixup=0.0,  # Отключили mixup
                copy_paste=0.0
            )

            print("Обучение успешно завершено!")
            return True

        except Exception as e:
            print(f"Ошибка обучения: {e}")
            return False

    def evaluate_model(self):
        """Оценка модели на тестовой выборке"""
        print("Оценка модели на тестовой выборке...")

        if self.model is None:
            print("Модель не инициализирована")
            return None

        try:
            best_model_path = os.path.join(self.script_dir, 'runs', 'detect',
                                           f'people_detection_{self.timestamp}', 'weights', 'best.pt')
            if os.path.exists(best_model_path):
                self.model = YOLO(best_model_path)
                print("Загружена лучшая модель после обучения")

                # Копируем лучшую модель в папку результатов
                shutil.copy2(best_model_path, os.path.join(self.output_dir, 'models', 'best.pt'))

            # Очистка памяти перед валидацией
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            metrics = self.model.val(
                data=os.path.join(self.yolo_dataset_path, 'data.yaml'),
                split='test',
                device=self.device,
                batch=8,  # Уменьшили batch для валидации
                workers=2
            )

            print("Результаты оценки:")
            print(f"  mAP50-95: {metrics.box.map:.4f}")
            print(f"  mAP50: {metrics.box.map50:.4f}")
            print(f"  mAP75: {metrics.box.map75:.4f}")
            print(f"  Precision: {metrics.box.precision:.4f}")
            print(f"  Recall: {metrics.box.recall:.4f}")

            self.save_metrics_to_file(metrics)

            return metrics

        except Exception as e:
            print(f"Ошибка оценки: {e}")
            return None

    def save_metrics_to_file(self, metrics):
        """Сохранение метрик в файл"""
        metrics_data = {
            'mAP50-95': float(metrics.box.map),
            'mAP50': float(metrics.box.map50),
            'mAP75': float(metrics.box.map75),
            'precision': float(metrics.box.precision),
            'recall': float(metrics.box.recall),
            'training_time': self.timestamp,
            'model': 'YOLOv11n',
            'dataset': 'People Detection',
            'epochs': 50,
            'image_size': 640,
            'batch_size': 12
        }

        metrics_path = os.path.join(self.output_dir, 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=4, ensure_ascii=False)

        print(f"Метрики сохранены: {metrics_path}")

    def save_training_plots(self):
        """Сохранение графиков обучения"""
        if self.results is None:
            print("Нет результатов обучения для построения графиков")
            return

        try:
            results_dict = self.results.results_dict

            if results_dict:
                # Графики потерь
                plt.figure(figsize=(15, 10))

                metrics = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss',
                           'val/box_loss', 'val/cls_loss', 'val/dfl_loss']

                for i, metric in enumerate(metrics, 1):
                    plt.subplot(2, 3, i)
                    if metric in results_dict:
                        plt.plot(results_dict[metric])
                        plt.title(metric.replace('/', ' ').title())
                        plt.xlabel('Эпоха')
                        plt.ylabel('Потери')
                        plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'graphs', 'training_losses.png'),
                            dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()

                # Графики метрик
                plt.figure(figsize=(15, 5))

                metrics_to_plot = [
                    ('metrics/precision(B)', 'Precision'),
                    ('metrics/recall(B)', 'Recall'),
                    ('metrics/mAP50(B)', 'mAP50'),
                    ('metrics/mAP50-95(B)', 'mAP50-95')
                ]

                for i, (metric, title) in enumerate(metrics_to_plot, 1):
                    plt.subplot(1, 4, i)
                    if metric in results_dict:
                        plt.plot(results_dict[metric])
                        plt.title(title)
                        plt.xlabel('Эпоха')
                        plt.ylabel('Значение')
                        plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'graphs', 'training_metrics.png'),
                            dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()

                print(f"Графики обучения сохранены: {os.path.join(self.output_dir, 'graphs')}")
            else:
                print("Нет данных истории обучения")

        except Exception as e:
            print(f"Ошибка сохранения графиков: {e}")


class DetectionVisualizer:
    def __init__(self, model_path, output_dir):
        self.model = YOLO(model_path)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def test_on_dataset(self, dataset_path, num_images=5):
        """Тестирование на изображениях из датасета"""
        test_images_dir = os.path.join(dataset_path, 'test', 'images')

        if not os.path.exists(test_images_dir):
            print(f"Директория тестовых изображений не найдена: {test_images_dir}")
            return

        image_files = glob.glob(os.path.join(test_images_dir, "*.jpg"))
        image_files = image_files[:num_images]

        print(f"Тестирование на {len(image_files)} изображениях из датасета...")

        for i, image_path in enumerate(image_files):
            try:
                # Очистка памяти перед каждой обработкой
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                results = self.model.predict(image_path, conf=0.25, imgsz=640)
                plotted_image = results[0].plot()
                output_path = os.path.join(self.output_dir, f'dataset_test_{i + 1:03d}.jpg')
                cv2.imwrite(output_path, plotted_image)

                num_detections = len(results[0].boxes) if results[0].boxes else 0
                print(f"Изображение {i + 1}/{len(image_files)}: обнаружено {num_detections} человек")

            except Exception as e:
                print(f"Ошибка обработки {image_path}: {e}")


def main():
    """Основная функция"""
    print(" ЗАПУСК ПРОГРАММЫ ОБНАРУЖЕНИЯ ЛЮДЕЙ")
    print("=" * 50)

    # Очистка памяти перед началом
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1. Инициализация
    detector = PeopleDetectorTrainer(dataset_path='lab3')
    detector.setup_environment()

    # 2. Подготовка датасета
    if not detector.prepare_dataset():
        print(" Не удалось подготовить датасет. Проверьте структуру папок.")
        return

    detector.analyze_dataset()

    # 3. Инициализация модели
    if not detector.setup_model():
        return

    # 4. Обучение модели
    print("\n БЕЗОПАСНЫЕ ПАРАМЕТРЫ ОБУЧЕНИЯ:")
    print("  - Размер изображения: 640x640")
    print("  - Batch size: 12 (уменьшен для стабильности)")
    print("  - Количество эпох: 50 (уменьшено)")
    print("  - Workers: 4 (уменьшено)")
    print("  - Mosaic: отключен (экономия памяти)")
    print("  - Mixup: отключен (экономия памяти)")
    print("  - Лимит памяти GPU: 70%")

    if not detector.train_model(epochs=50, imgsz=640, batch=12):
        return

    # 5. Оценка модели
    print("\n ОЦЕНКА МОДЕЛИ...")
    metrics = detector.evaluate_model()

    # 6. Сохранение графиков
    print("\n СОХРАНЕНИЕ ГРАФИКОВ...")
    detector.save_training_plots()

    # 7. Тестирование детектора
    print("\n ТЕСТИРОВАНИЕ ДЕТЕКТОРА...")
    best_model_path = os.path.join(detector.output_dir, 'models', 'best.pt')

    if os.path.exists(best_model_path):
        visualizer = DetectionVisualizer(best_model_path,
                                         os.path.join(detector.output_dir, 'detections'))

        # Тестирование на датасете
        visualizer.test_on_dataset(detector.yolo_dataset_path, num_images=5)

        print(f"\n ВСЕ ЗАДАЧИ ВЫПОЛНЕНЫ УСПЕШНО!")
        print("=" * 50)
        print(f" РЕЗУЛЬТАТЫ СОХРАНЕНЫ В: {detector.output_dir}")

        if metrics:
            print(f"\n ФИНАЛЬНЫЕ МЕТРИКИ МОДЕЛИ:")
            print(f"   mAP50: {metrics.box.map50:.3f}")
            print(f"   mAP50-95: {metrics.box.map:.3f}")
            print(f"   Precision: {metrics.box.precision:.3f}")
            print(f"   Recall: {metrics.box.recall:.3f}")
    else:
        print(" Лучшая модель не найдена")


if __name__ == "__main__":
    main()