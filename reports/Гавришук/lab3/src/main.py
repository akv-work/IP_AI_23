import os
import sys
import argparse
import subprocess
import shutil
import json
from pathlib import Path
from typing import List
import requests
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def pip_install(pkg: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

def ensure_yolov10_installed():
    try:
        import ultralytics  # type: ignore
    except Exception:
        print("[INFO] ultralytics/yolov10 не найден — ставим из репозитория YOLOv10 (THU-MIG)...")
        pip_install("git+https://github.com/THU-MIG/yolov10.git")

    try:
        from ultralytics import YOLO  # type: ignore
        return True
    except Exception as e:
        print("[ERROR] Не удалось импортировать ultralytics.YOLO после установки:", e)
        return False

def create_data_yaml(data_dir: str, names: List[str]=["cat"]) -> str:
    data_dir = Path(data_dir)
    out = data_dir / "data.yaml"
    content = {
        "path": str(data_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(names),
        "names": names
    }
    with open(out, "w", encoding="utf-8") as f:
        yaml_text = json.dumps(content, indent=2)
        f.write(f"path: {content['path']}\n")
        f.write(f"train: {content['train']}\n")
        f.write(f"val: {content['val']}\n")
        f.write(f"test: {content['test']}\n")
        f.write(f"nc: {content['nc']}\n")
        f.write("names:\n")
        for i, n in enumerate(names):
            f.write(f"  {i}: {n}\n")
    print(f"[INFO] Создан {out}")
    return str(out)

def run_training(data_yaml: str, epochs: int=50, batch: int=16, imgsz: int=640, project: str="runs/train", name: str="yolov10s_cats"):
    ok = ensure_yolov10_installed()
    if not ok:
        raise RuntimeError("Требуется установить yolov10/ultralytics. Смотрите логи выше.")

    from ultralytics import YOLO

    model_cfgs = ["yolov10s.yaml", "yolov10s.pt", "yolov10n.yaml", "yolov10n.pt"]
    model_source = None
    for m in model_cfgs:
        try:
            model = YOLO(m)
            model_source = m
            break
        except Exception:
            continue

    if model_source is None:
        raise RuntimeError("Не найдено заранее доступных конфигов/весов yolov10s. Установите репозиторий yolov10 и убедитесь, что файлы yolov10s.yaml / yolov10s.pt доступны.")

    print(f"[INFO] Использую модель/конфиг: {model_source}")
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, project=project, name=name)
    print("[INFO] Обучение завершено (или прервано).")

def run_validation(weights_path: str, data_yaml: str, imgsz: int=640):
    from ultralytics import YOLO
    model = YOLO(weights_path)
    print("[INFO] Запускаю валидацию на тестовом наборе...")
    res = model.val(data=data_yaml, imgsz=imgsz)
    print("[INFO] Результат валидации:")
    print(res)
    return res

def download_image(url: str, out_dir: str="tmp_images") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    local_name = Path(out_dir) / Path(url.split("?")[0].split("/")[-1])
    if not local_name.exists():
        print(f"[INFO] Скачиваем {url} -> {local_name}")
        r = requests.get(url, stream=True, timeout=15)
        r.raise_for_status()
        with open(local_name, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return str(local_name)

def visualize_predictions(weights_path: str, image_urls: List[str], imgsz: int=640, conf: float=0.25, save_dir: str="predictions"):
    from ultralytics import YOLO
    model = YOLO(weights_path)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for url in image_urls:
        try:
            img_path = download_image(url)
        except Exception as e:
            print("[WARN] Не удалось скачать", url, e)
            continue
        results = model(img_path, imgsz=imgsz, conf=conf)
        r = results[0]
        pil = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(pil)
        try:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
        except Exception:
            boxes = []
            scores = []
            classes = []
        names = getattr(r, "names", None) or getattr(model, "names", None) or {}
        font = ImageFont.load_default()
        for (xy, sc, cl) in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, xy.tolist())
            label = f"{names.get(cl, str(cl))} {sc:.2f}"
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            text_w, text_h = font.getsize(label)
            draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill="red")
            draw.text((x1, y1 - text_h), label, fill="white", font=font)
        out_path = Path(save_dir) / ("pred_" + Path(img_path).name)
        pil.save(out_path)
        print(f"[INFO] Сохранено предсказание: {out_path}")

def check_dataset_structure(data_dir: str) -> bool:
    data_dir = Path(data_dir)
    req = [
        data_dir / "train" / "images",
        data_dir / "train" / "labels",
        data_dir / "valid" / "images",
        data_dir / "valid" / "labels",
    ]
    ok = True
    for p in req:
        if not p.exists():
            print(f"[WARN] Отсутствует {p}")
            ok = False
    if ok:
        print("[INFO] Структура датасета выглядит корректно.")
    else:
        print("[ERROR] Проверьте, что вы скачали Roboflow датасет и распаковали в:", data_dir)
    return ok

def main():
    import torch
    print("CUDA доступна:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Используется видеокарта:", torch.cuda.get_device_name(0))
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./datasets/cats", help="путь к распакованному датасету (YOLO формат)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--train-only", action="store_true", help="только тренировать (без валидации/визуализации)")
    parser.add_argument("--weights", type=str, default="", help="если указать путь к весам для валидации/визуализации")
    parser.add_argument("--visualize-urls", type=str, nargs="*", default=[], help="URL-изображения для отрисовки предсказаний")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not check_dataset_structure(data_dir):
        print("\n[INSTR] Скачайте датасет YOLO с Roboflow и распакуйте в", data_dir)
        print("Roboflow dataset page:", "https://universe.roboflow.com/mohamed-traore-2ekkp/cats-n9b87/dataset/3")
        sys.exit(1)

    data_yaml = create_data_yaml(data_dir, names=["cat"])

    run_training(data_yaml, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)

    run_dir = Path("runs/train")
    last_run = None
    if run_dir.exists():
        runs = sorted([d for d in run_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime)
        if runs:
            last_run = runs[-1]
    if last_run:
        candidate_weights = last_run / "weights" / "best.pt"
        if candidate_weights.exists():
            best_weights = str(candidate_weights)
        else:
            pts = list(last_run.rglob("*.pt"))
            best_weights = str(pts[-1]) if pts else ""
    else:
        best_weights = ""

    if best_weights:
        print("[INFO] Найдены веса для валидации/визуализации:", best_weights)
    else:
        print("[WARN] Не найдены веса автоматически. Укажите --weights путь к .pt если хотите запустить валидацию/визуализацию.")
        if args.train_only:
            print("[DONE] Завершено (только тренировка).")
            return

    if best_weights:
        run_validation(best_weights, data_yaml, imgsz=args.imgsz)

    if args.visualize_urls:
        weights_for_vis = args.weights if args.weights else best_weights
        if not weights_for_vis:
            print("[WARN] Нет весов для визуализации предсказаний.")
        else:
            visualize_predictions(weights_for_vis, args.visualize_urls, imgsz=args.imgsz)

if __name__ == "__main__":
    main()
