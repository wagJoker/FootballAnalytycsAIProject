# src/evaluate_yolo.py
"""Оценка обученных YOLO моделей."""
from pathlib import Path
from typing import Optional

from ultralytics import YOLO


def evaluate_model(
    weights_path: str | Path,
    run_name: str,
    dataset_config: Optional[str] = None,
    imgsz: int = 640,
) -> dict:
    """Оценка модели на валидационном наборе.
    
    Args:
        weights_path: Путь к файлу с весами модели
        run_name: Имя запуска для сохранения результатов
        dataset_config: Путь к конфигурации датасета (опционально)
        imgsz: Размер изображения для инференса
        
    Returns:
        Словарь с метриками модели
    """
    weights_path = Path(weights_path)
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    print(f"[INFO] Loading model from: {weights_path}")
    model = YOLO(str(weights_path))
    
    print(f"[INFO] Evaluating {weights_path} on validation set")
    
    val_kwargs = {
        "imgsz": imgsz,
        "project": "runs_kaggle",
        "name": f"{run_name}-eval",
        "plots": True,
        "save_json": True,
    }
    
    if dataset_config:
        val_kwargs["data"] = dataset_config
    
    results = model.val(**val_kwargs)
    
    # Вывод основных метрик
    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        print("\n[INFO] Evaluation metrics:")
        print(f"  mAP50: {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"  mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        print(f"  Precision: {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
        print(f"  Recall: {metrics.get('metrics/recall(B)', 'N/A'):.4f}")
    
    return results.results_dict if hasattr(results, "results_dict") else {}


def main() -> None:
    """Основная функция для оценки модели."""
    default_weights = Path("runs_kaggle/detect/yolo8n_baseline/weights/best.pt")
    
    if not default_weights.exists():
        print(f"[ERROR] {default_weights} not found.")
        print("[INFO] Train model first using: python src/train_yolo.py")
        return
    
    try:
        evaluate_model(
            weights_path=default_weights,
            run_name="yolo8n_baseline",
            dataset_config="configs/football_kaggle.yaml",
        )
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}")


if __name__ == "__main__":
    main()

