from typing import Tuple
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def model_and_loss(in_features: int, num_classes: int = 2) -> Tuple[nn.Module, nn.Module]:
    model = MLP(in_features, num_classes)
    criterion = nn.CrossEntropyLoss()
    return model, criterion