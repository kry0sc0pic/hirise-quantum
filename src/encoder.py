import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """
    Shared-weight Siamese CNN encoder.

    Architecture:
        (1, 64, 64)
        → Conv2d(1, 32, 3, pad=1) + ReLU + MaxPool2d(2) → (32, 32, 32)
        → Conv2d(32, 64, 3, pad=1) + ReLU + MaxPool2d(2) → (64, 16, 16)
        → Flatten → Linear(16384, 128) + ReLU
        → Linear(128, n_qubits) → L2 normalize
    """

    def __init__(self, n_qubits: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, n_qubits)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # (B, 32, 32, 32)
        x = self.pool(F.relu(self.conv2(x)))   # (B, 64, 16, 16)
        x = x.flatten(1)                        # (B, 16384)
        x = F.relu(self.fc1(x))                 # (B, 128)
        x = self.fc2(x)                          # (B, n_qubits)
        return F.normalize(x, p=2, dim=-1)       # unit sphere in R^n_qubits
