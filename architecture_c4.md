# C4 Architecture - Football AI Analytics System

This document describes the software architecture of the Football AI Analytics System using the C4 model (Context, Containers, Components, Code).

## Level 1: System Context Diagram

The System Context diagram showing how the Football AI Analytics System fits into the world around it.

```mermaid
C4Context
    title System Context Diagram for Football AI Analytics System

    Person(analyst, "Sports Analyst", "A football analyst or coach wanting insights from match footage.")
    
    System(football_ai, "Football AI Analytics System", "Provides player tracking, metrics analysis, and AI-generated match reports.")

    System_Ext(kaggle, "Kaggle Datasets", "Source of training images and labels.")
    System_Ext(video_source, "Video Feeds", "Raw match footage.")

    Rel(analyst, football_ai, "Configures experiments, Views reports", "CLI / Files")
    Rel(football_ai, kaggle, "Downloads training data")
    Rel(football_ai, video_source, "Ingests video")
    Rel(football_ai, analyst, "Generates tracking CSVs, plots, and text summaries")
```

## Level 2: Container Diagram

The Container diagram shows the high-level technical building blocks.

```mermaid
C4Container
    title Container Diagram for Football AI Analytics System

    Person(analyst, "Sports Analyst", "Uses CLI to run pipelines")

    Container_Boundary(c1, "Football AI System") {
        Container(cv_pipeline, "CV Pipeline", "Python, YOLOv8", "Handles object detection and tracking on video.")
        Container(rag_engine, "RAG Engine", "Python, Pandas", "Retrieves and filters match statistics to build context.")
        Container(quantum_llm, "Quantum Hybrid LLM", "Python, PyTorch, PennyLane", "Experimental module for generating text using hybrid quantum-classical models.")
        Container(logging, "Logging Service", "Python Logging", "Centralized logging for all modules.")
    }

    ContainerDb(fs, "File System", "NTFS/Ext4", "Stores raw videos, Kaggle datasets, models (.pt), metrics (.csv), and logs.")

    Rel(analyst, cv_pipeline, "Runs training/tracking", "CLI")
    Rel(analyst, quantum_llm, "Runs RAG/Generation", "CLI")

    Rel(cv_pipeline, fs, "Reads images/video, Writes tracks/metrics")
    Rel(rag_engine, fs, "Reads match metrics (CSV)")
    Rel(quantum_llm, rag_engine, "Uses for context retrieval")
    Rel(quantum_llm, fs, "Reads/Writes model weights")
    
    Rel(cv_pipeline, logging, "Logs events")
    Rel(rag_engine, logging, "Logs events")
    Rel(quantum_llm, logging, "Logs events")
```

## Level 3: Component Diagram

The Component diagram shows the internals of the key containers.

### CV Pipeline Components
```mermaid
C4Component
    title Component Diagram - CV Pipeline

    Container(cv_pipeline, "CV Pipeline", "Python Scripts")

    Component(trainer, "YOLO Trainer", "src/train_yolo.py", "Fine-tunes YOLO models on football datasets.")
    Component(tracker, "Video Tracker", "src/track_video.py", "Runs inference on video and applies ByteTrack.")
    Component(evaluator, "Evaluator", "src/evaluate_yolo.py", "Calculates mAP and other metrics.")
    Component(dataset_debug, "Dataset Debugger", "src/debug_dataset.py", "Validates and sanitizes image data.")

    ContainerDb(models_dir, "Models Directory", "models/", "Stores .pt files")
    ContainerDb(data_dir, "Data Directory", "data/", "Stores images/labels")

    Rel(trainer, data_dir, "Reads training data")
    Rel(trainer, models_dir, "Saves trained weights")
    Rel(tracker, models_dir, "Loads weights")
    Rel(tracker, data_dir, "Reads video")
    Rel(evaluator, models_dir, "Loads weights")
```

### Quantum LLM & RAG Components
```mermaid
C4Component
    title Component Diagram - Quantum LLM & RAG

    Container(quantum_llm, "Quantum Hybrid LLM", "FootballQuantumLLM/")
    Container(rag_engine, "RAG Module", "rag/")

    Component(q_layer, "Quantum Layer", "quantum_layer.py", "PennyLane circuit or Mock simulation.")
    Component(hybrid_gpt, "Hybrid GPT", "model_hybrid.py", "Classical Transformer + Quantum Layer.")
    Component(trainer_llm, "LLM Trainer", "train.py", "Training loop for Hybrid GPT.")
    Component(rag_agent, "RAG Agent", "rag_agent.py", "Orchestrator for retrieval and generation.")

    Component(loader, "Loader", "rag/loader.py", "Loads CSV data.")
    Component(retriever, "Retriever", "rag/retriever.py", "Filters data by query params.")
    
    ContainerDb(metrics_csv, "Metrics CSV", "runs_kaggle/tracking/metrics.csv", "Match stats")

    Rel(trainer_llm, hybrid_gpt, "Optimizes")
    Rel(hybrid_gpt, q_layer, "Uses for embedding/features")
    Rel(rag_agent, hybrid_gpt, "Uses for text generation")
    Rel(rag_agent, retriever, "Retrieves context")
    Rel(retriever, loader, "Uses")
    Rel(loader, metrics_csv, "Reads")
```

## Level 4: Code Diagram (Classes)

The Class diagram detailing the `FootballQuantumLLM` and `RAG` implementation.

```mermaid
classDiagram
    class QuantumLayer {
        +n_qubits: int
        +n_layers: int
        +q_params: Parameter
        +forward(x: Tensor) Tensor
    }
    
    class GPTLanguageModel {
        +token_embedding_table: Embedding
        +position_embedding_table: Embedding
        +blocks: Sequential
        +lm_head: Linear
        +forward(idx: Tensor, targets: Tensor) Tensor
        +generate(idx: Tensor, max_new_tokens: int) Tensor
    }
    
    class HybridGPT {
        +quantum_layer: QuantumLayer
        +token_embedding_table: Embedding
        +projection: Linear
        +forward(idx: Tensor, targets: Tensor) Tensor
    }
    
    class QuantumRAGAgent {
        +model: HybridGPT
        +df: DataFrame
        +generate_report(query_team: str) str
    }
    
    class FootballRAGService {
        +data: DataFrame
        +answer_query(query: str) str
    }

    HybridGPT --> QuantumLayer : Uses (Composition)
    QuantumRAGAgent --> HybridGPT : Uses
    HybridGPT --|> GPTLanguageModel : Architectural Variant (Decoder-Only)
```
