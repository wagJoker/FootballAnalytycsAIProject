# Football YOLO Analytics

System for football player detection and tracking on video using YOLO models (YOLOv8, YOLOv12).

## 🎯 Features

- 📥 Automated Kaggle dataset download and preparation
- 🚀 Training of multiple model configurations (A/B testing)
- 🎬 Player tracking on video using ByteTrack
- 📊 Visualization of tracking metrics and statistics
- 🔧 Flexible configuration via YAML files

## 📋 Requirements

- Python 3.8+
- CUDA (optional, for GPU acceleration)
- Kaggle API credentials (for dataset download)

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd football-yolo-analytics
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
# OR for Windows:
install_deps.bat
```

4. Configure Kaggle API (for dataset download):
   - Create an account on [Kaggle](https://www.kaggle.com/)
   - Download API credentials: `~/.kaggle/kaggle.json`
   - Or use environment variables

## 📖 Usage

### 1. Download Dataset

```bash
python src/download_kaggle_dataset.py
```

This will create the structure:
```
data/football_kaggle/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### 2. Train Models

Train all experiments from `configs/ab_experiments.yaml`:

```bash
python src/train_yolo.py
```

Or train a single experiment (modify code in `train_yolo.py`).

### 3. Evaluate Model

Evaluate a specific model:

```bash
python src/evaluate_yolo.py
```

### 4. A/B Testing

Run evaluation for all trained models and compare results:

```bash
python src/ab_test_runner.py
```

Results are saved in `runs_kaggle/ab_results.csv`.

### 5. Video Tracking

Track football players on video:

```bash
python src/track_video.py
```

**Important:** Before running, ensure that:
- The model is trained (file `runs_kaggle/detect/yolo8n_baseline/weights/best.pt` exists)
- The video file exists at `data/video/match1.mp4`

Or modify paths in `src/track_video.py`.

### 6. Visualize Results

Plot A/B testing results:

```bash
python src/utils_viz.py
```

## 📁 Project Structure

```
football-yolo-analytics/
├── configs/              # Configuration files
│   ├── football_kaggle.yaml
│   ├── ab_experiments.yaml
│   └── rag.yaml
├── data/                 # Data (created automatically)
│   └── football_kaggle/
├── datasets/             # Generated datasets
│   └── dataset.pt        # Quantum LLM dataset
├── docs/                 # C4 Documentation
│   ├── C4_Context.md
│   ├── C4_Container.md
│   └── C4_Component.md
├── src/                  # CV Source Code
│   ├── config_loader.py
│   ├── download_kaggle_dataset.py
│   ├── train_yolo.py
│   ├── evaluate_yolo.py
│   ├── ab_test_runner.py
│   ├── track_video.py
│   └── utils_viz.py
├── rag/                  # RAG Module
│   ├── loader.py
│   ├── retriever.py
│   └── rag_service.py
├── FootballQuantumLLM/   # Quantum LLM (Experimental)
│   ├── data_gen.py
│   ├── model_classical.py
│   ├── model_hybrid.py
│   ├── quantum_layer.py
│   ├── train.py
│   ├── rag_agent.py
│   └── logger.py
├── logs/                 # Logs
│   └── football_quantum.log
├── runs_kaggle/          # Results (created automatically)
│   ├── detect/           # Trained models
│   ├── tracking/         # Tracking results
│   └── ab_results.csv    # A/B testing results
├── .github/
│   └── workflows/
│       └── ci.yml        # CI/CD Pipeline
├── requirements.txt
├── architecture_c4.md    # C4 Architecture Diagram
├── install_deps.bat      # Windows Dependency Installer
├── README.md
└── .gitignore
```

## 🧠 New Modules

### 1. RAG (Retrieval-Augmented Generation)

Module for searching match statistics and generating answers.

**Run Demo:**
```bash
python src/rag_demo.py
```

### 2. Experimental FootballQuantumLLM

Hybrid Quantum-Classical LLM for generating match reports (Educational Prototype).

**Usage:**

1. Generate Data:
   ```bash
   python -m FootballQuantumLLM.data_gen
   ```
   (Generates `datasets/dataset.pt`)

2. Train Models (Classical vs Hybrid):
   ```bash
   python -m FootballQuantumLLM.train
   ```
   (Uses `datasets/dataset.pt`)
3. Run RAG Agent based on Quantum LLM:
   ```bash
   python -m FootballQuantumLLM.rag_agent
   ```

**Logging:**
All operations are logged to the `logs/` directory.

### 3. C4 Architecture

The system architecture is documented in C4 format. 
👉 **[View Architecture Documentation (architecture_c4.md)](architecture_c4.md)**


## ⚙️ Configuration

### Dataset Configuration (`configs/football_kaggle.yaml`)

```yaml
path: ../data/football_kaggle
train: images/train
val: images/val
nc: 1
names:
  0: player
```

### Experiment Configuration (`configs/ab_experiments.yaml`)

```yaml
experiments:
  - name: yolo8n_baseline
    model: yolov8n.pt
    imgsz: 640
    epochs: 50
    batch: 16
    lr0: 0.001
```

## 📊 Metrics

The system calculates the following metrics:

- **mAP50**: Mean Average Precision at IoU=0.5
- **mAP50-95**: Mean Average Precision at IoU=0.5:0.95
- **Precision**: Detection Precision
- **Recall**: Detection Recall
- **F1**: F1-Score

## 🔧 Development

### Running Tests

```bash
# Check imports
python -c "from src.config_loader import load_yaml"

# Linting (if flake8, black, isort are installed)
flake8 src/
black src/
isort src/
```

### CI/CD

The project uses GitHub Actions for automatic code checks on push and pull request.

## 📝 License

[Specify License]

## 🤝 Contribution

Contributions are welcome! Please create an issue or pull request.

## 📚 Additional Resources

- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [Kaggle Dataset](https://www.kaggle.com/datasets/adilshamim8/football-players-detection)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)

## ⚠️ Known Issues

- Kaggle API credentials are required for dataset download.
- YOLOv12 might not be available in some Ultralytics versions (remove the experiment from config if needed).
- Tracking requires significant computational resources for large videos.

## 🆘 Support

If you encounter issues:

1. Check that all dependencies are installed.
2. Ensure the dataset is downloaded and the structure is correct.
3. Check file paths in configurations.
4. Create an issue describing the problem.
