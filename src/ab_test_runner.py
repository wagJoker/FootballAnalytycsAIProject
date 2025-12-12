# src/ab_test_runner.py
"""Запуск A/B тестирования различных конфигураций моделей."""
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from ultralytics import YOLO

# Добавляем путь к src для импортов при прямом запуске
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

from config_loader import load_yaml

CONFIG_AB = "configs/ab_experiments.yaml"
CONFIG_DATASET = "configs/football_kaggle.yaml"


def eval_single(weights_path: str | Path, exp_name: str, dataset_config: str) -> dict[str, Any]:
    """Оценка одной модели.
    
    Args:
        weights_path: Путь к файлу с весами
        exp_name: Имя эксперимента
        dataset_config: Путь к конфигурации датасета
        
    Returns:
        Словарь с метриками эксперимента
    """
    model = YOLO(str(weights_path))
    
    results = model.val(
        data=dataset_config,
        imgsz=640,
        plots=False,
        verbose=False,
    )
    
    metrics = results.results_dict if hasattr(results, "results_dict") else {}
    
    return {
        "experiment": exp_name,
        "weights": str(weights_path),
        "mAP50": metrics.get("metrics/mAP50(B)", 0.0),
        "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0.0),
        "precision": metrics.get("metrics/precision(B)", 0.0),
        "recall": metrics.get("metrics/recall(B)", 0.0),
        "f1": metrics.get("metrics/f1(B)", 0.0),
    }


def main() -> None:
    """Основная функция для запуска A/B тестирования."""
    if not Path(CONFIG_AB).exists():
        print(f"[ERROR] Config file not found: {CONFIG_AB}")
        return
    
    if not Path(CONFIG_DATASET).exists():
        print(f"[ERROR] Dataset config not found: {CONFIG_DATASET}")
        return
    
    ab_cfg = load_yaml(CONFIG_AB)
    
    if "experiments" not in ab_cfg:
        print("[ERROR] 'experiments' key not found in config file")
        return
    
    experiments = ab_cfg["experiments"]
    rows = []
    
    print(f"[INFO] Starting A/B testing for {len(experiments)} experiments\n")
    
    for i, exp in enumerate(experiments, 1):
        name = exp["name"]
        weights = Path(f"runs_kaggle/detect/{name}/weights/best.pt")
        
        if not weights.exists():
            print(f"[WARN] [{i}/{len(experiments)}] Weights not found for {name}, skipping.")
            continue
        
        print(f"[INFO] [{i}/{len(experiments)}] Evaluating {name}...")
        try:
            row = eval_single(weights, name, CONFIG_DATASET)
            rows.append(row)
            print(f"  ✓ mAP50-95: {row['mAP50-95']:.4f}")
        except Exception as e:
            print(f"  ✗ Error evaluating {name}: {e}")
    
    if not rows:
        print("\n[ERROR] No experiments evaluated successfully.")
        return
    
    # Создание DataFrame и сортировка
    df = pd.DataFrame(rows).sort_values(by="mAP50-95", ascending=False)
    
    # Вывод результатов
    print(f"\n{'='*80}")
    print("A/B Testing Results:")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"{'='*80}\n")
    
    # Сохранение результатов
    output_dir = Path("runs_kaggle")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ab_results.csv"
    
    df.to_csv(output_path, index=False)
    print(f"[INFO] Results saved to: {output_path}")
    
    # Вывод лучшей модели
    best = df.iloc[0]
    print(f"\n[INFO] Best model: {best['experiment']} (mAP50-95: {best['mAP50-95']:.4f})")


if __name__ == "__main__":
    main()

