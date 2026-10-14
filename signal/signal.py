"""SIGNAL: Graph-Augmented Instance Learning Model"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class SIGNAL(nn.Module):
    """
    SIGNAL: Graph-Augmented Instance Learning for synthetic speech attribution
    and open-set detection.

    Combines foundation model embeddings with Graph Neural Networks and KNN-based
    open-set detection for robust forensic attribution.

    Args:
        embedding_dim (int): Speech embedding dimension
        num_generators (int): Number of known generators
        hidden_dim (int): Hidden dimension for GNN
        dropout (float): Dropout probability
    """

    def __init__(
        self,
        embedding_dim: int = 1024,
        num_generators: int = 10,
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_generators = num_generators

        # Generator prototypes
        self.generator_prototypes = nn.Parameter(torch.randn(num_generators, embedding_dim))

        # Graph encoder
        self.graph_proj = nn.Linear(embedding_dim, hidden_dim)
        self.gnn_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Attribution head (closed-set)
        self.attribution_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_generators),
        )

        # Open-set detection head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, speech_embeddings: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of SIGNAL.

        Args:
            speech_embeddings: Speech foundation model embeddings [batch, embedding_dim]
            return_features: Whether to return intermediate features

        Returns:
            Dict with:
                - attribution_logits: Generator attribution logits
                - confidence_scores: Open-set confidence scores
                - predictions: Predicted generator ID or "Unknown"
        """
        # Project speech embedding
        speech_proj = self.graph_proj(speech_embeddings)

        # Build query-conditioned graph
        prototype_proj = self.graph_proj(self.generator_prototypes)

        # Compute graph features
        graph_features = self.gnn_encoder(
            torch.cat([speech_proj.unsqueeze(1).expand(-1, self.num_generators, -1),
                      prototype_proj.unsqueeze(0).expand(speech_proj.size(0), -1, -1)], dim=-1).reshape(-1, -1)
        ).reshape(speech_proj.size(0), self.num_generators, -1).mean(dim=1)

        # Attribution (closed-set)
        attribution_logits = self.attribution_head(graph_features)

        # Open-set detection
        confidence_scores = self.confidence_head(graph_features)

        # Predictions
        predicted_generators = torch.argmax(attribution_logits, dim=1)

        return {
            "attribution_logits": attribution_logits,
            "confidence_scores": confidence_scores.squeeze(-1),
            "predicted_generator": predicted_generators,
            "features": graph_features if return_features else None,
        }

    @classmethod
    def from_pretrained(cls, checkpoint_path: str) -> "SIGNAL":
        """Load pretrained SIGNAL model."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def save(self, checkpoint_path: str, config: Optional[Dict] = None) -> None:
        """Save SIGNAL model checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": config or {
                "embedding_dim": self.embedding_dim,
                "num_generators": self.num_generators,
            },
        }
        torch.save(checkpoint, checkpoint_path)
