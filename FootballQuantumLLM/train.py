# FootballQuantumLLM/train.py
import torch
import torch.nn as nn
from FootballQuantumLLM.data_gen import VOCAB_SIZE, encode, decode, generate_dataset
from FootballQuantumLLM.model_classical import GPTLanguageModel
from FootballQuantumLLM.model_hybrid import HybridGPT
import time
from FootballQuantumLLM.logger import setup_logger

logger = setup_logger()

# Hyperparameters
BATCH_SIZE = 16
BLOCK_SIZE = 32
MAX_ITERS = 50   # Short training for prototype
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_INTERVAL = 10
N_EMBD = 32      # Small for speed/quantum simulation
N_HEAD = 2
N_LAYER = 2

def get_batch(data):
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

@torch.no_grad()
def estimate_loss(model, data):
    out = {}
    model.eval()
    losses = torch.zeros(10)
    for k in range(10):
        X, Y = get_batch(data)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()

def train_model(model_cls, name, data):
    logger.info(f"--- Training {name} ---")
    model = model_cls(VOCAB_SIZE, n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER, block_size=BLOCK_SIZE)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    start_time = time.time()
    for iter in range(MAX_ITERS):
        xb, yb = get_batch(data)
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iter % EVAL_INTERVAL == 0:
            logger.info(f"Iter {iter}: loss {loss.item():.4f}")
            
    end_time = time.time()
    logger.info(f"Training finished in {end_time - start_time:.2f}s")
    
    # Generate sample
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    logger.info(f"{name} generated sample:")
    print(decode(model.generate(context, max_new_tokens=50)[0].tolist()))
    
    return model, loss.item()

def main():
    # 1. Prepare Data
    if not torch.cuda.is_available():
        logger.info("Running on CPU.")
    else:
        logger.info("Running on CUDA.")
        
    dataset_path = "datasets/dataset.pt"
    try:
        data = torch.load(dataset_path)
    except FileNotFoundError:
        logger.info("Generating data...")
        data = generate_dataset(output_path=dataset_path)
    
    # Train Classical
    classical_model, c_loss = train_model(GPTLanguageModel, "Classical GPT", data)
    torch.save(classical_model.state_dict(), "FootballQuantumLLM/classical_model.pth")
    
    # Train Hybrid
    hybrid_model, h_loss = train_model(HybridGPT, "Hybrid Quantum GPT", data)
    torch.save(hybrid_model.state_dict(), "FootballQuantumLLM/hybrid_model.pth")
    
    logger.info("--- Comparison ---")
    logger.info(f"Classical Final Loss: {c_loss:.4f}")
    logger.info(f"Hybrid Final Loss:    {h_loss:.4f}")

if __name__ == "__main__":
    main()
