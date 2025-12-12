# src/debug_dataset.py
from pathlib import Path
import cv2

# 1. Проверка пустых label-файлов
def check_empty_labels():
    labels_dir = Path("data/football_kaggle/labels/train")
    empty = []

    for txt in labels_dir.glob("*.txt"):
        content = txt.read_text().strip()
        if not content:
            empty.append(txt.name)

    print("Empty label files:", len(empty))
    print("First empty labels:", empty[:10])

# 2. Проверка совпадения имён картинок и разметки
def check_name_match():
    img_dir = Path("data/football_kaggle/images/train")
    lbl_dir = Path("data/football_kaggle/labels/train")

    imgs = {p.stem for p in img_dir.glob("*.jpg")}
    lbls = {p.stem for p in lbl_dir.glob("*.txt")}

    print("Images:", len(imgs), "Labels:", len(lbls))
    print("No label for (first 20):", list(imgs - lbls)[:20])
    print("No image for (first 20):", list(lbls - imgs)[:20])

# 3. Проверка битых изображений
def validate_and_sanitize_dataset(dataset_path: Path) -> None:
    """Проверяет изображения в датасете и изолирует битые файлы.
    
    Args:
        dataset_path: Путь к корневой директории датасета (должна содержать images/train, images/val и т.д.)
    """
    print(f"[INFO] Validating dataset at: {dataset_path}")
    
    images_root = dataset_path / "images"
    labels_root = dataset_path / "labels"
    
    if not images_root.exists():
        print(f"[WARN] Images directory not found: {images_root}")
        return

    # Рекурсивный поиск всех изображений
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in extensions:
        image_files.extend(list(images_root.rglob(ext)))
        
    print(f"[INFO] Found {len(image_files)} images. Scanning for corruption...")
    
    broken_count = 0
    for img_path in image_files:
        is_broken = False
        try:
            # Пытаемся прочитать изображение через OpenCV
            img = cv2.imread(str(img_path))
            if img is None:
                is_broken = True
            else:
                # Дополнительная проверка: иногда imread возвращает None без исключения, 
                # но если вернул массив, проверим размеры
                if img.shape[0] == 0 or img.shape[1] == 0:
                     is_broken = True
        except Exception:
            is_broken = True
            
        if is_broken:
            broken_count += 1
            print(f"[WARN] Corruped image found: {img_path}")
            
            # 1. Переименовываем битое изображение
            new_img_path = img_path.with_suffix(img_path.suffix + ".broken")
            try:
                img_path.rename(new_img_path)
                print(f"       Renamed to: {new_img_path.name}")
            except Exception as e:
                print(f"       [ERROR] Failed to rename image: {e}")
                continue

            # 2. Удаляем/переименовываем соответствующий файл метки (если есть)
            # Структура: images/train/img.jpg <-> labels/train/img.txt
            # Нужно вычислить путь к label файлу, предполагая зеркальную структуру
            try:
                rel_path = img_path.relative_to(images_root) # train/img.jpg
                label_rel_path = rel_path.with_suffix(".txt") # train/img.txt
                label_path = labels_root / label_rel_path
                
                if label_path.exists():
                    new_label_path = label_path.with_suffix(".txt.broken")
                    label_path.rename(new_label_path)
                    print(f"       Renamed label to: {new_label_path.name}")
            except Exception as e:
                print(f"       [WARN] Could not process label for broken image: {e}")

    if broken_count == 0:
        print("[INFO] No broken images found. Dataset is clean.")
    else:
        print(f"[INFO] Found and sanitized {broken_count} broken images.")

def check_empty_labels():
    labels_dir = Path("data/football_kaggle/labels/train")
    # ... (оставляем старые функции вспомогательными или удаляем если не нужны, но лучше оставить для ручного дебага)
    # Для краткости заменим реализацию main, а старые функции оставим
    pass 

if __name__ == "__main__":
    # Для теста запускаем на дефолтном пути
    default_path = Path("inputs/football_kaggle") # Путь может отличаться
    # Лучше использовать путь из конфига, но для простого запуска укажем хардкод или аргумент
    import sys
    if len(sys.argv) > 1:
        validate_and_sanitize_dataset(Path(sys.argv[1]))
    else:
        validate_and_sanitize_dataset(Path("data/football_kaggle"))
