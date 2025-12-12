# C4: Container

## Containers

### 1. CLI Training Service

**Описание:** Сервис для обучения YOLO моделей на футбольном датасете.

**Скрипты:**
- `src/train_yolo.py` – обучение моделей по конфигурациям из `ab_experiments.yaml`
- `src/ab_test_runner.py` – запуск A/B тестирования и сравнение метрик

**Зависимости:**
- Ultralytics (YOLO)
- Kaggle датасет (через kagglehub)
- YAML конфигурации

**Технологии:**
- Python 3.8+
- PyTorch (через Ultralytics)
- YAML для конфигураций

---

### 2. Dataset Downloader

**Описание:** Утилита для скачивания и подготовки датасета в формате YOLO.

**Скрипты:**
- `src/download_kaggle_dataset.py` – скачивание датасета Kaggle и организация структуры

**Зависимости:**
- kagglehub
- Kaggle API credentials

**Технологии:**
- Python 3.8+
- Kaggle Hub API

---

### 3. Evaluation Service

**Описание:** Сервис для оценки обученных моделей.

**Скрипты:**
- `src/evaluate_yolo.py` – оценка модели на валидационном наборе

**Зависимости:**
- Обученные модели (`.pt` файлы)
- Валидационный датасет

**Технологии:**
- Ultralytics
- Метрики: mAP50, mAP50-95, Precision, Recall, F1

---

### 4. Video Tracking Service

**Описание:** Сервис для трекинга футболистов на видео.

**Скрипты:**
- `src/track_video.py` – трекинг объектов на видео с сохранением результатов в CSV

**Зависимости:**
- Обученная модель детекции
- Входное видео

**Технологии:**
- Ultralytics (YOLO + ByteTrack)
- OpenCV
- CSV для экспорта треков

---

### 5. Visualization Service

**Описание:** Утилиты для визуализации результатов экспериментов.

**Скрипты:**
- `src/utils_viz.py` – построение графиков метрик A/B тестирования и статистики трекинга

**Зависимости:**
- Результаты A/B тестирования (CSV)
- Треки (CSV)

**Технологии:**
- Matplotlib
- Pandas

---

### 6. Configuration Management

**Описание:** Управление конфигурациями датасетов и экспериментов.

**Файлы:**
- `configs/football_kaggle.yaml` – конфигурация датасета YOLO
- `configs/ab_experiments.yaml` – конфигурации A/B экспериментов

**Технологии:**
- YAML
- `src/config_loader.py` – загрузчик конфигураций

---

## Data Flow

1. **Dataset Preparation:**
   ```
   Kaggle → download_kaggle_dataset.py → data/football_kaggle/
   ```

2. **Training:**
   ```
   ab_experiments.yaml → train_yolo.py → runs_kaggle/detect/{experiment}/weights/best.pt
   ```

3. **Evaluation:**
   ```
   best.pt → evaluate_yolo.py / ab_test_runner.py → ab_results.csv
   ```

4. **Tracking:**
   ```
   video.mp4 + best.pt → track_video.py → tracks.csv + annotated_video.mp4
   ```

5. **Visualization:**
   ```
   ab_results.csv → utils_viz.py → plots (matplotlib)
   ```

