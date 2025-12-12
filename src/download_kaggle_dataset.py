# src/download_kaggle_dataset.py
"""Скачивание и подготовка датасета Kaggle для обучения YOLO."""
import shutil
from pathlib import Path

import kagglehub

DATA_ROOT = Path("data/football_kaggle")
KAGGLE_DATASET = "adilshamim8/football-players-detection"


def copy_tree(src: Path, dst: Path) -> None:
    """Копирует все файлы из src в dst.
    
    Args:
        src: Исходная директория
        dst: Целевая директория
    """
    if not src.exists():
        print(f"[WARN] Source path not found: {src}")
        return
    
    dst.mkdir(parents=True, exist_ok=True)
    
    files_copied = 0
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
            files_copied += 1
    
    print(f"[INFO] Copied {files_copied} files from {src} to {dst}")


def main() -> None:
    """Основная функция для скачивания и подготовки датасета."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Downloading Kaggle dataset: {KAGGLE_DATASET}")
    
    try:
        # Скачиваем датасет целиком
        ds_path = Path(
            kagglehub.dataset_download(KAGGLE_DATASET)
        )
        print(f"[INFO] Kaggle dataset downloaded to: {ds_path}")
    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}")
        print("[INFO] Make sure you have kagglehub configured with your credentials")
        return
    
    # Посмотри содержимое ds_path после первого запуска
    # Ниже пример для структуры:
    # ds_path/
    #   images/train, images/val
    #   labels/train, labels/val
    src_images_train = ds_path / "images" / "train"
    src_images_val = ds_path / "images" / "val"
    src_labels_train = ds_path / "labels" / "train"
    src_labels_val = ds_path / "labels" / "val"
    
    dst_images_train = DATA_ROOT / "images" / "train"
    dst_images_val = DATA_ROOT / "images" / "val"
    dst_labels_train = DATA_ROOT / "labels" / "train"
    dst_labels_val = DATA_ROOT / "labels" / "val"
    
    print("[INFO] Copying files to YOLO structure...")
    copy_tree(src_images_train, dst_images_train)
    copy_tree(src_images_val, dst_images_val)
    copy_tree(src_labels_train, dst_labels_train)
    copy_tree(src_labels_val, dst_labels_val)
    
    # Проверка структуры
    print("\n[INFO] Verifying dataset structure...")
    required_dirs = [
        dst_images_train, dst_images_val,
        dst_labels_train, dst_labels_val
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        if dir_path.exists() and any(dir_path.iterdir()):
            file_count = len(list(dir_path.iterdir()))
            print(f"  ✓ {dir_path}: {file_count} files")
        else:
            print(f"  ✗ {dir_path}: missing or empty")
            all_ok = False
    
    if all_ok:
        print(f"\n[INFO] YOLO-style data prepared successfully under: {DATA_ROOT}")
    else:
        print(f"\n[WARN] Some directories are missing. Please check the dataset structure.")


if __name__ == "__main__":
    main()

