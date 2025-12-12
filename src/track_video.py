# src/track_video.py
"""Трекинг футболистов на видео с использованием обученной YOLO модели."""
from pathlib import Path
import csv
from typing import Optional

import cv2
from ultralytics import YOLO

MODEL_PATH = "runs_kaggle/detect/yolo8n_baseline/weights/best.pt"
VIDEO_PATH = "data/video/match1.mp4"
OUT_DIR = Path("runs_kaggle/tracking")
TRACK_CSV = OUT_DIR / "tracks.csv"
CLASS_PLAYER = 0  # из football_kaggle.yaml


def track_video(
    model_path: str | Path,
    video_path: str | Path,
    output_dir: Path = OUT_DIR,
    output_csv: Optional[Path] = None,
    conf_threshold: float = 0.4,
    iou_threshold: float = 0.5,
    imgsz: int = 640,
    class_filter: Optional[int] = CLASS_PLAYER,
) -> Path:
    """Трекинг объектов на видео.
    
    Args:
        model_path: Путь к файлу с весами модели
        video_path: Путь к входному видео
        output_dir: Директория для сохранения результатов
        output_csv: Путь к CSV файлу с треками (опционально)
        conf_threshold: Порог уверенности детекции
        iou_threshold: Порог IoU для NMS
        imgsz: Размер изображения для инференса
        class_filter: Класс для фильтрации (None = все классы)
        
    Returns:
        Путь к сохраненному CSV файлу с треками
    """
    model_path = Path(model_path)
    video_path = Path(video_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_csv is None:
        video_name = video_path.stem
        output_csv = output_dir / f"{video_name}_tracks.csv"
    
    print(f"[INFO] Loading model from: {model_path}")
    model = YOLO(str(model_path))
    
    print(f"[INFO] Processing video: {video_path}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Confidence threshold: {conf_threshold}")
    print(f"[INFO] IoU threshold: {iou_threshold}")
    
    # Трекинг
    results = model.track(
        source=str(video_path),
        imgsz=imgsz,
        conf=conf_threshold,
        iou=iou_threshold,
        show=False,
        save=True,
        save_vid=True,
        project=str(output_dir),
        name=video_path.stem,
        tracker="bytetrack.yaml",
        verbose=True,
    )
    
    # Сохранение треков в CSV
    print(f"[INFO] Saving tracks to CSV: {output_csv}")
    
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "track_id", "class", "x_center", "y_center", "w", "h",
            "confidence", "x1", "y1", "x2", "y2"
        ])
        
        frame_id = -1
        total_detections = 0
        
        for r in results:
            frame_id += 1
            boxes = r.boxes
            
            if boxes is None:
                continue
            
            for box in boxes:
                cls = int(box.cls.item())
                
                # Фильтрация по классу
                if class_filter is not None and cls != class_filter:
                    continue
                
                # Получение track_id
                tid = int(box.id.item()) if box.id is not None else -1
                
                # Координаты bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x_c = (x1 + x2) / 2
                y_c = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                
                # Уверенность
                conf = float(box.conf.item()) if box.conf is not None else 0.0
                
                writer.writerow([
                    frame_id, tid, cls, x_c, y_c, w, h,
                    conf, x1, y1, x2, y2
                ])
                total_detections += 1
    
    print(f"[INFO] Tracking completed:")
    print(f"  Total frames: {frame_id + 1}")
    print(f"  Total detections: {total_detections}")
    print(f"  CSV saved to: {output_csv}")
    
    return output_csv


def main() -> None:
    """Основная функция для трекинга видео."""
    try:
        track_video(
            model_path=MODEL_PATH,
            video_path=VIDEO_PATH,
            output_csv=TRACK_CSV,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("\n[INFO] Make sure:")
        print(f"  1. Model is trained: python src/train_yolo.py")
        print(f"  2. Video file exists at: {VIDEO_PATH}")
    except Exception as e:
        print(f"[ERROR] Tracking failed: {e}")


if __name__ == "__main__":
    main()

