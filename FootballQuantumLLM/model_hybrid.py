# FootballQuantumLLM/model_hybrid.py
import torch
import torch.nn as nn
from FootballQuantumLLM.model_classical import GPTLanguageModel
from FootballQuantumLLM.quantum_layer import QuantumLayer

class HybridGPT(GPTLanguageModel):
    def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=4, block_size=128, dropout=0.0):
        super().__init__(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
        
        # WE REPLACE or AUGMENT one part of the network.
        # Strategy: Add a Quantum "bottleneck" or "adaptor" after the first embedding,
        # or replaced a feed-forward block.
        # Here: We'll add a Quantum Layer in parallel to the first token embedding 
        # to "enrich" the representation.
        
        self.n_qubits = 4 
        # Project n_embd down to n_qubits for quantum processing
        self.q_proj_in = nn.Linear(n_embd, self.n_qubits)
        self.quantum_layer = QuantumLayer(n_qubits=self.n_qubits)
        # Project back up
        self.q_proj_out = nn.Linear(self.n_qubits, n_embd)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Standard Embedding
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb 
        
        # --- QUANTUM ENRICHMENT ---
        # We process the embedding through a quantum layer
        # Since quantum simulation is slow, we might only do this for the CLS token or aggregate,
        # but here we do it for every token (Warning: VERY SLOW if T is large and not using vectorization)
        # To make it feasible for training, we flatten (B*T, C)
        
        x_flat = x.view(-1, x.size(-1)) # (B*T, C)
        q_in = torch.tanh(self.q_proj_in(x_flat)) # Map to [-1, 1] for rotation encoding usually
        q_out = self.quantum_layer(q_in)   # (B*T, n_qubits)
        q_feat = self.q_proj_out(q_out)    # (B*T, C)
        
        # Add quantum features to classical embeddings (Hybrid Residual Connection)
        x_flat = x_flat + q_feat
        x = x_flat.view(B, T, -1)
        # ---------------------------

        x = self.blocks(x) 
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss
