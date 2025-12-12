# FootballQuantumLLM/data_gen.py
import torch
import pandas as pd
from pathlib import Path
import random
from FootballQuantumLLM.logger import setup_logger

logger = setup_logger()

# Use a simple character-level tokenizer for this educational prototype
# In production, use BPE/BytePair encoding

CHARS = "".join(sorted(list(set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-=|\n"
))))
STOI = {ch: i for i, ch in enumerate(CHARS)}
ITOS = {i: ch for i, ch in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)

def encode(text: str) -> list[int]:
    return [STOI.get(c, 0) for c in text]  # 0 as unknown/padding if needed

def decode(tokens: list[int]) -> str:
    return "".join([ITOS.get(t, "?") for t in tokens])

def generate_dataset(csv_path: str | Path = "../runs_kaggle/tracking/metrics.csv", output_path: str = "datasets/dataset.pt"):
    """
    Simulates generating a dataset of (context, target) pairs.
    Context: Semi-structured metrics
    Target: Natural language summary
    """
    path = Path(csv_path)
    if not path.exists():
        logger.warning(f"{path} not found. using synthetic data.")
        # Create synthetic dataframe if file missing
        data = {
            "match_id": [101, 101, 102, 102] * 50,
            "team": ["Home", "Away", "Home", "Away"] * 50,
            "xG": [1.2, 0.8, 2.5, 0.1] * 50,
            "shots": [10, 8, 15, 2] * 50
        }
        df = pd.DataFrame(data)
    else:
        logger.info(f"Loading data from {path}")
        df = pd.read_csv(path)

    # Generate synthetic pairs
    contexts = []
    
    summary_templates = [
        "{team} showed strong dominance with {xG} expected goals.",
        "{team} struggled effectively but managed {shots} shots.",
        "An intense match for {team} resulted in xG of {xG}.",
    ]

    for _, row in df.iterrows():
        # Build Context
        ctx_str = f"Match:{row.get('match_id',0)}|Team:{row.get('team','?')} | xG:{row.get('xG',0.0)} | Shots:{row.get('shots',0)}"
        
        # Build Target (synthetic "ground truth" for LM training)
        t_temp = random.choice(summary_templates)
        tgt_str = t_temp.format(
            team=row.get('team', "Unknown"),
            xG=row.get('xG', 0.0),
            shots=row.get('shots', 0)
        )
        
        # Format: <Context> -> <Target>
        # We model it as a single sequence: Context \n Target
        full_text = f"{ctx_str}\nSummary: {tgt_str}\n"
        
        contexts.append(full_text)

    # Tokenize
    all_data = "".join(contexts)
    tensor_data = torch.tensor(encode(all_data), dtype=torch.long)
    
    logger.info(f"Generated dataset with {len(tensor_data)} tokens.")
    logger.info(f"Vocab size: {VOCAB_SIZE}")
    # logger.debug(f"Example:\n{contexts[0]}") # Debug level might be too verbose for console by default, but keeping it simple
    print(f"Example:\n{contexts[0]}") # Keep print for immediate visibility in this demo
    
    torch.save(tensor_data, output_path)
    return tensor_data

if __name__ == "__main__":
    generate_dataset()
