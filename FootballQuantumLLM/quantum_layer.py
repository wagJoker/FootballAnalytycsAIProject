# FootballQuantumLLM/quantum_layer.py
import torch
import torch.nn as nn
from FootballQuantumLLM.logger import setup_logger

logger = setup_logger()

# Try to import PennyLane, otherwise use a mock
try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False
    logger.warning("PennyLane not found. Using simulated Quantum Layer.")

class QuantumLayer(nn.Module):
    """
    A Quantum Layer that encodes input features into a quantum circuit
    and measures the expectation values.
    """
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        if HAS_PENNYLANE:
            self.dev = qml.device("default.qubit", wires=n_qubits)
            
            @qml.qnode(self.dev, interface="torch")
            def _circuit(inputs, weights):
                # Encoding Scheme (Amplitude or Angle Encoding)
                # Here we use Angle Encoding for simplicity
                for i in range(n_qubits):
                    qml.RX(inputs[i % len(inputs)], wires=i)
                
                # Variational Layers
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                
                # Measurement
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
            self.qnode = _circuit
            # Weight shape for StronglyEntanglingLayers: (n_layers, n_qubits, 3)
            weight_shapes = {"weights": (n_layers, n_qubits, 3)}
            # We use qml.qnn.TorchLayer to wrap it easily
            try:
                self.qlayer = qml.qnn.TorchLayer(self.qnode, weight_shapes)
            except AttributeError:
                 # Fallback for older versions or if qnn missing
                self.qlayer = None 
                self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))

        else:
            # MOCK Layer: Simulates a "quantum" transformation using random unitary-like operations
            # x -> sin(Wx + b) to mimic non-linearity
            self.mock_weight = nn.Parameter(torch.randn(n_qubits, n_qubits))

    def forward(self, x):
        """
        Input x: (batch_size, n_features)
        In this demo, we project n_features down to n_qubits if needed, or repeat.
        """
        if HAS_PENNYLANE and getattr(self, "qlayer", None):
            return self.qlayer(x)
        elif HAS_PENNYLANE:
             # Manual Forward if TorchLayer failed (unlikely)
             res = [self.qnode(x[i], self.weights) for i in range(x.shape[0])]
             return torch.stack(res)
        else:
            # Mock Quantum Behavior
            # Nonlinear transformation mimicking interference
            # In real quantum layer, outputs result from measurements in [-1, 1] usually
            out = torch.matmul(x, self.mock_weight)
            out = torch.sin(out * 3.14159) # Periodic encoding
            return out
