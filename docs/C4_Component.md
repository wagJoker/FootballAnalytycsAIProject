# C4: Component

## Component Diagram

### Training Pipeline Components

#### 1. ConfigLoader
**Файл:** `src/config_loader.py`

**Ответственность:**
- Загрузка YAML конфигурационных файлов
- Валидация структуры конфигураций

**Интерфейсы:**
- `load_yaml(path: str | Path) -> dict[str, Any]`

**Зависимости:**
- PyYAML

---

#### 2. DatasetDownloader
**Файл:** `src/download_kaggle_dataset.py`

**Ответственность:**
- Скачивание датасета из Kaggle
- Организация файлов в структуру YOLO (images/train, images/val, labels/train, labels/val)
- Валидация структуры после копирования

**Интерфейсы:**
- `main() -> None`
- `copy_tree(src: Path, dst: Path) -> None`

**Зависимости:**
- kagglehub
- shutil, pathlib

---

#### 3. ModelTrainer
**Файл:** `src/train_yolo.py`

**Ответственность:**
- Загрузка конфигураций экспериментов
- Инициализация YOLO моделей
- Обучение моделей с заданными гиперпараметрами
- Сохранение весов и метрик

**Интерфейсы:**
- `train_experiment(exp: dict, dataset_config: str) -> Path | None`
- `main() -> None`

**Зависимости:**
- Ultralytics (YOLO)
- ConfigLoader

**Взаимодействие:**
- Читает `configs/ab_experiments.yaml`
- Читает `configs/football_kaggle.yaml`
- Сохраняет результаты в `runs_kaggle/detect/{name}/`

---

#### 4. ModelEvaluator
**Файл:** `src/evaluate_yolo.py`

**Ответственность:**
- Оценка обученной модели на валидационном наборе
- Вычисление метрик (mAP50, mAP50-95, Precision, Recall)
- Сохранение результатов оценки

**Интерфейсы:**
- `evaluate_model(weights_path, run_name, dataset_config, imgsz) -> dict`
- `main() -> None`

**Зависимости:**
- Ultralytics (YOLO)
- Валидационный датасет

---

#### 5. ABTestRunner
**Файл:** `src/ab_test_runner.py`

**Ответственность:**
- Запуск оценки всех экспериментов из конфигурации
- Сбор метрик в единую таблицу
- Сравнение результатов и определение лучшей модели
- Экспорт результатов в CSV

**Интерфейсы:**
- `eval_single(weights_path, exp_name, dataset_config) -> dict`
- `main() -> None`

**Зависимости:**
- ModelEvaluator (через YOLO)
- Pandas для обработки данных
- ConfigLoader

**Взаимодействие:**
- Читает `configs/ab_experiments.yaml`
- Читает веса из `runs_kaggle/detect/{name}/weights/best.pt`
- Сохраняет результаты в `runs_kaggle/ab_results.csv`

---

#### 6. VideoTracker
**Файл:** `src/track_video.py`

**Ответственность:**
- Загрузка обученной модели
- Обработка видео кадр за кадром
- Детекция и трекинг объектов (ByteTrack)
- Экспорт треков в CSV формат
- Сохранение аннотированного видео

**Интерфейсы:**
- `track_video(model_path, video_path, output_dir, ...) -> Path`
- `main() -> None`

**Зависимости:**
- Ultralytics (YOLO + ByteTrack)
- OpenCV
- CSV writer

**Выходные данные:**
- CSV с колонками: frame, track_id, class, x_center, y_center, w, h, confidence, x1, y1, x2, y2
- Аннотированное видео

---

#### 7. VisualizationUtils
**Файл:** `src/utils_viz.py`

**Ответственность:**
- Визуализация результатов A/B тестирования
- Построение графиков метрик
- Визуализация статистики трекинга
- Экспорт графиков в файлы

**Интерфейсы:**
- `plot_ab_results(csv_path, save_path) -> None`
- `plot_tracking_statistics(csv_path) -> None`

**Зависимости:**
- Matplotlib
- Pandas

---

## Component Interactions

```
┌─────────────────┐
│ ConfigLoader    │
└────────┬────────┘
         │
         ├─────────────────┬──────────────────┐
         │                 │                  │
         ▼                 ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Dataset      │  │ ModelTrainer │  │ ABTestRunner │
│ Downloader   │  └──────┬───────┘  └──────┬───────┘
└──────────────┘         │                  │
                         │                  │
                         ▼                  ▼
                  ┌──────────────┐  ┌──────────────┐
                  │ YOLO Models  │  │ ModelEvaluator│
                  │ (Ultralytics)│  └──────┬───────┘
                  └──────┬───────┘         │
                         │                 │
                         ▼                 ▼
                  ┌──────────────┐  ┌──────────────┐
                  │ VideoTracker │  │ Visualization│
                  └──────────────┘  └──────────────┘
```

## Data Storage

- **Configs:** `configs/*.yaml`
- **Dataset:** `data/football_kaggle/`
- **Model Weights:** `runs_kaggle/detect/{experiment}/weights/`
- **Results:** `runs_kaggle/ab_results.csv`
- **Tracking:** `runs_kaggle/tracking/{video_name}_tracks.csv`
- **Videos:** `runs_kaggle/tracking/{video_name}/`

