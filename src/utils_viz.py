# src/utils_viz.py
"""Утилиты для визуализации результатов экспериментов."""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_ab_results(csv_path: str | Path = "runs_kaggle/ab_results.csv", save_path: Optional[Path] = None) -> None:
    """Визуализация результатов A/B тестирования.
    
    Args:
        csv_path: Путь к CSV файлу с результатами
        save_path: Путь для сохранения графика (опционально)
    """
    path = Path(csv_path)
    if not path.exists():
        print(f"[ERROR] {csv_path} not found")
        print("[INFO] Run A/B testing first: python src/ab_test_runner.py")
        return
    
    df = pd.read_csv(path)
    
    if df.empty:
        print("[ERROR] CSV file is empty")
        return
    
    # Создание фигуры с несколькими подграфиками
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("A/B Experiments - Football Players Detection", fontsize=16, fontweight="bold")
    
    # График 1: mAP50-95
    axes[0, 0].bar(df["experiment"], df["mAP50-95"], color="steelblue")
    axes[0, 0].set_ylabel("mAP50-95", fontsize=12)
    axes[0, 0].set_title("Mean Average Precision (mAP50-95)", fontsize=12, fontweight="bold")
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].grid(axis="y", alpha=0.3)
    
    # График 2: mAP50
    axes[0, 1].bar(df["experiment"], df["mAP50"], color="forestgreen")
    axes[0, 1].set_ylabel("mAP50", fontsize=12)
    axes[0, 1].set_title("Mean Average Precision (mAP50)", fontsize=12, fontweight="bold")
    axes[0, 1].tick_params(axis="x", rotation=20)
    axes[0, 1].grid(axis="y", alpha=0.3)
    
    # График 3: Precision vs Recall
    axes[1, 0].scatter(df["precision"], df["recall"], s=100, alpha=0.6, c=range(len(df)), cmap="viridis")
    for i, row in df.iterrows():
        axes[1, 0].annotate(row["experiment"], (row["precision"], row["recall"]), 
                           fontsize=8, alpha=0.7)
    axes[1, 0].set_xlabel("Precision", fontsize=12)
    axes[1, 0].set_ylabel("Recall", fontsize=12)
    axes[1, 0].set_title("Precision vs Recall", fontsize=12, fontweight="bold")
    axes[1, 0].grid(alpha=0.3)
    
    # График 4: F1 Score (если есть)
    if "f1" in df.columns:
        axes[1, 1].bar(df["experiment"], df["f1"], color="coral")
        axes[1, 1].set_ylabel("F1 Score", fontsize=12)
        axes[1, 1].set_title("F1 Score", fontsize=12, fontweight="bold")
    else:
        # Комбинированный график Precision и Recall
        x = range(len(df))
        width = 0.35
        axes[1, 1].bar([i - width/2 for i in x], df["precision"], width, label="Precision", color="skyblue")
        axes[1, 1].bar([i + width/2 for i in x], df["recall"], width, label="Recall", color="lightcoral")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(df["experiment"], rotation=20)
        axes[1, 1].set_ylabel("Score", fontsize=12)
        axes[1, 1].set_title("Precision and Recall", fontsize=12, fontweight="bold")
        axes[1, 1].legend()
        axes[1, 1].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Plot saved to: {save_path}")
    else:
        plt.show()


def plot_tracking_statistics(csv_path: str | Path) -> None:
    """Визуализация статистики трекинга из CSV файла.
    
    Args:
        csv_path: Путь к CSV файлу с треками
    """
    path = Path(csv_path)
    if not path.exists():
        print(f"[ERROR] {csv_path} not found")
        return
    
    df = pd.read_csv(path)
    
    if df.empty:
        print("[ERROR] CSV file is empty")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Tracking Statistics", fontsize=16, fontweight="bold")
    
    # График 1: Количество детекций по кадрам
    detections_per_frame = df.groupby("frame").size()
    axes[0, 0].plot(detections_per_frame.index, detections_per_frame.values, linewidth=1.5)
    axes[0, 0].set_xlabel("Frame", fontsize=12)
    axes[0, 0].set_ylabel("Number of Detections", fontsize=12)
    axes[0, 0].set_title("Detections per Frame", fontsize=12, fontweight="bold")
    axes[0, 0].grid(alpha=0.3)
    
    # График 2: Распределение track_id
    track_counts = df["track_id"].value_counts().sort_index()
    axes[0, 1].bar(range(len(track_counts)), track_counts.values, color="steelblue")
    axes[0, 1].set_xlabel("Track ID", fontsize=12)
    axes[0, 1].set_ylabel("Number of Detections", fontsize=12)
    axes[0, 1].set_title("Detections per Track ID", fontsize=12, fontweight="bold")
    axes[0, 1].grid(axis="y", alpha=0.3)
    
    # График 3: Распределение уверенности
    if "confidence" in df.columns:
        axes[1, 0].hist(df["confidence"], bins=50, color="forestgreen", alpha=0.7, edgecolor="black")
        axes[1, 0].set_xlabel("Confidence", fontsize=12)
        axes[1, 0].set_ylabel("Frequency", fontsize=12)
        axes[1, 0].set_title("Confidence Distribution", fontsize=12, fontweight="bold")
        axes[1, 0].grid(alpha=0.3)
    
    # График 4: Траектории (если есть координаты)
    if "x_center" in df.columns and "y_center" in df.columns:
        # Показываем траектории для первых 10 track_id
        top_tracks = df["track_id"].value_counts().head(10).index
        for tid in top_tracks:
            track_data = df[df["track_id"] == tid]
            axes[1, 1].plot(track_data["x_center"], track_data["y_center"], 
                          alpha=0.6, linewidth=1.5, label=f"Track {tid}")
        axes[1, 1].set_xlabel("X Center", fontsize=12)
        axes[1, 1].set_ylabel("Y Center", fontsize=12)
        axes[1, 1].set_title("Player Trajectories (Top 10)", fontsize=12, fontweight="bold")
        axes[1, 1].legend(fontsize=8, ncol=2)
        axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Пример использования
    plot_ab_results()
    # plot_tracking_statistics("runs_kaggle/tracking/tracks.csv")

