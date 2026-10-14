"""Graph Neural Network encoder for generator relationship reasoning"""

import torch
import torch.nn as nn


class GraphEncoder(nn.Module):
    """Graph Neural Network for reasoning about generator relationships"""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, node_features, edge_features=None):
        """Encode graph using GNN"""
        return self.encoder(node_features)
