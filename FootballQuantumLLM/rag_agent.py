# FootballQuantumLLM/rag_agent.py
import torch
import sys
import os
from pathlib import Path

# Add project root to path to verify imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from FootballQuantumLLM.model_hybrid import HybridGPT
from FootballQuantumLLM.data_gen import VOCAB_SIZE, encode, decode
from rag.retriever import simple_filter
from rag.loader import load_matches_csv
from FootballQuantumLLM.logger import setup_logger

logger = setup_logger()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class QuantumRAGAgent:
    def __init__(self, model_path="FootballQuantumLLM/hybrid_model.pth", csv_path="runs_kaggle/tracking/metrics.csv"):
        # Load Data
        self.df = load_matches_csv(csv_path)
        
        # Load Model
        self.model = HybridGPT(VOCAB_SIZE, n_embd=32, n_head=2, n_layer=2, block_size=32)
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            logger.info(f"Loaded hybrid model from {model_path}")
        else:
            logger.warning("Model weights found. Using random init.")
        self.model.to(DEVICE)
        self.model.eval()

    def generate_report(self, match_id, team=None):
        # 1. Retrieve
        relevant_rows = simple_filter(self.df, match_id=match_id, team=team, top_n=5)
        
        # 2. Build Context
        context_str = ""
        for _, r in relevant_rows.iterrows():
             context_str += f"Match:{r.get('match_id')}|Team:{r.get('team')} | xG:{r.get('xG')} | Shots:{r.get('shots')}\n"
        
        print(f"Retrieved Context:\n{context_str}")
        
        # 3. Generate
        # We start generation with the context
        start_ids = encode(context_str)
        x = torch.tensor(start_ids, dtype=torch.long, device=DEVICE)[None, ...] # (1, T)
        
        # Cap context length to block_size if needed (simplification)
        if x.shape[1] > 32:
             x = x[:, -32:]

        # Create "summary" prompt feel
        prompt = "\nSummary:"
        p_ids = encode(prompt)
        x = torch.cat([x, torch.tensor(p_ids, dtype=torch.long, device=DEVICE)[None, ...]], dim=1)

        print("Generating...")
        y = self.model.generate(x, max_new_tokens=50)
        
        output_text = decode(y[0].tolist())
        return output_text

def main():
    agent = QuantumRAGAgent()
    report = agent.generate_report(match_id=101, team="Home")
    print(f"\n--- Generated Report ---\n{report}")

if __name__ == "__main__":
    main()
