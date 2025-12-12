# src/train_yolo.py
"""Обучение YOLO моделей для детекции футболистов."""
import sys
from pathlib import Path
from typing import Any

from ultralytics import YOLO

# Добавляем путь к src для импортов при прямом запуске
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

from config_loader import load_yaml

CONFIG_DATASET = "configs/football_kaggle.yaml"
CONFIG_AB = "configs/ab_experiments.yaml"


def train_experiment(exp: dict[str, Any], dataset_config: str = CONFIG_DATASET) -> Path | None:
    """Обучение одной конфигурации модели (A/B эксперимент).
    
    Args:
        exp: Словарь с параметрами эксперимента
        dataset_config: Путь к конфигурации датасета
        
    Returns:
        Путь к файлу с весами модели или None в случае ошибки
    """
    name = exp["name"]
    model_name = exp["model"]
    
    print(f"\n{'='*60}")
    print(f"[INFO] Training experiment: {name}")
    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] Epochs: {exp.get('epochs', 50)}")
    print(f"[INFO] Image size: {exp.get('imgsz', 640)}")
    print(f"[INFO] Batch size: {exp.get('batch', 16)}")
    print(f"[INFO] Learning rate: {exp.get('lr0', 1e-3)}")
    print(f"{'='*60}\n")
    
    try:
        # Проверка наличия модели (для локальных моделей)
        if model_name.endswith('.pt') and not Path(model_name).exists():
            # Проверяем, не является ли это стандартной моделью YOLO
            standard_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
            if model_name not in standard_models:
                print(f"[ERROR] Model file not found: {model_name}")
                print(f"[INFO] Skipping experiment {name}")
                print(f"[INFO] Note: {model_name} may not be available in this Ultralytics version")
                return None
        
        # Попытка загрузки модели
        try:
            model = YOLO(model_name)
        except Exception as model_error:
            print(f"[ERROR] Failed to load model {model_name}: {model_error}")
            print(f"[INFO] Skipping experiment {name}")
            return None
        
        model.train(
            data=dataset_config,
            epochs=exp.get("epochs", 50),
            imgsz=exp.get("imgsz", 640),
            batch=exp.get("batch", 16),
            lr0=exp.get("lr0", 1e-3),
            optimizer=exp.get("optimizer", "AdamW"),
            project="runs_kaggle",
            name=name,
            val=True,
            save=True,
            plots=True,
        )
        
        weights_path = Path(f"runs_kaggle/detect/{name}/weights/best.pt")
        if weights_path.exists():
            print(f"\n[INFO] ✓ Finished experiment {name}")
            print(f"[INFO] Weights saved to: {weights_path}")
            return weights_path
        else:
            print(f"\n[WARN] Training completed but weights not found at: {weights_path}")
            return None
            
    except Exception as e:
        print(f"\n[ERROR] Failed to train experiment {name}: {e}")
        return None


def check_dataset(dataset_config: str) -> bool:
    """Проверка наличия датасета.
    
    Args:
        dataset_config: Путь к конфигурации датасета
        
    Returns:
        True если датасет существует, False иначе
    """
    config_path = Path(dataset_config)
    if not config_path.exists():
        print(f"[ERROR] Dataset config not found: {dataset_config}")
        return False
    
    cfg = load_yaml(dataset_config)
    dataset_path_str = cfg.get("path", "")
    
    # Разрешаем путь относительно конфига (как это делает YOLO)
    if Path(dataset_path_str).is_absolute():
        dataset_path = Path(dataset_path_str)
    else:
        # Относительный путь разрешается относительно директории конфига
        dataset_path = (config_path.parent / dataset_path_str).resolve()
    
    if not dataset_path.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_path}")
        print(f"[INFO] Expected path: {dataset_path}")
        print(f"[INFO] Please run: python src/download_kaggle_dataset.py")
        return False
    
    # Проверка наличия train и val директорий
    train_path = dataset_path / cfg.get("train", "images/train")
    val_path = dataset_path / cfg.get("val", "images/val")
    
    if not train_path.exists() or not any(train_path.iterdir()):
        print(f"[ERROR] Training images not found: {train_path}")
        print(f"[INFO] Please run: python src/download_kaggle_dataset.py")
        return False
    
    if not val_path.exists() or not any(val_path.iterdir()):
        print(f"[ERROR] Validation images not found: {val_path}")
        print(f"[INFO] Please run: python src/download_kaggle_dataset.py")
        return False
    
    train_count = len(list(train_path.glob("*.jpg"))) + len(list(train_path.glob("*.png")))
    val_count = len(list(val_path.glob("*.jpg"))) + len(list(val_path.glob("*.png")))
    
    print(f"[INFO] Dataset found:")
    print(f"  Train images: {train_count}")
    print(f"  Val images: {val_count}")
    
    return True


def main() -> None:
    """Основная функция для обучения всех экспериментов."""
    # Проверка наличия конфигурационных файлов
    if not Path(CONFIG_AB).exists():
        print(f"[ERROR] Config file not found: {CONFIG_AB}")
        return
    
    if not Path(CONFIG_DATASET).exists():
        print(f"[ERROR] Dataset config not found: {CONFIG_DATASET}")
        return
    
    # Проверка наличия датасета
    print("[INFO] Checking dataset...")
    
    # Интеграция валидации датасета (Added by User Request)
    try:
        from src.debug_dataset import validate_and_sanitize_dataset
        # Получаем путь к датасету из конфига для валидации
        dataset_cfg = load_yaml(CONFIG_DATASET)
        dataset_path_str = dataset_cfg.get("path", "../data/football_kaggle")
        # Разрешаем путь
        if Path(dataset_path_str).is_absolute():
            ds_path = Path(dataset_path_str)
        else:
            ds_path = (Path(CONFIG_DATASET).parent / dataset_path_str).resolve()
            
        validate_and_sanitize_dataset(ds_path)
    except Exception as e:
        print(f"[WARN] Failed to run dataset validation: {e}")

    if not check_dataset(CONFIG_DATASET):
        return
    
    ab_cfg = load_yaml(CONFIG_AB)
    
    if "experiments" not in ab_cfg:
        print("[ERROR] 'experiments' key not found in config file")
        return
    
    experiments = ab_cfg["experiments"]
    print(f"\n[INFO] Found {len(experiments)} experiments to train\n")
    
    results = []
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Starting experiment: {exp['name']}")
        weights_path = train_experiment(exp)
        results.append((exp["name"], weights_path))
    
    # Итоговая сводка
    print(f"\n{'='*60}")
    print("[INFO] Training summary:")
    print(f"{'='*60}")
    for name, weights_path in results:
        status = "✓" if weights_path and weights_path.exists() else "✗"
        print(f"  {status} {name}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

