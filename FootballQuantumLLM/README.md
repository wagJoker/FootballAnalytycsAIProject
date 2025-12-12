# FootballQuantumLLM

Experimental project exploring **Hybrid Quantum-Classical LLMs** for football analytics.

## Overview
This project integrates specific match metrics (xG, Shots, etc.) with a generative language model. It benchmarks a standard "Classical" Mini-GPT against a "Hybrid" model that includes a Quantum Encoding Layer (simulated or real via PennyLane).

## Architecture

1.  **Classical Baseline**: Decoder-only Transformer (NanoGPT style).
2.  **Quantum Hybrid**:
    -   Injects a **Quantum Layer** (Parameterized Quantum Circuit) into the embedding or FFN block.
    -   Uses `PennyLane` for quantum simulation (Torch interface).
    -   **Fallback**: If PennyLane is absent, a mock nonlinear layer mimics quantum interference.
3.  **RAG**: Retrieves match data from CSV and uses the Hybrid LLM to generate summaries.

## Usage

1.  **Generate Data**:
    ```bash
    python -m FootballQuantumLLM.data_gen
    ```
2.  **Train Models**:
    ```bash
    python -m FootballQuantumLLM.train
    ```
3.  **Run RAG Agent**:
    ```bash
    python -m FootballQuantumLLM.rag_agent
    ```

## Logging
Logs are automatically saved to `logs/football_quantum.log` in the project root.

## Requirements
- `torch`
- `pandas`
- `pennylane` (optional, for real quantum simulation)

## Future Work
-   Use real quantum hardware (IBM Q, IonQ) via PennyLane plugins.
-   Encoder-Decoder architecture for better sequence-to-sequence tasks.
-   Advanced Quantum Attention mechanisms.
