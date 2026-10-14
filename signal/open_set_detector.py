"""KNN-based open-set detection module"""

import torch
import torch.nn as nn


class OpenSetDetector(nn.Module):
    """KNN-based detector for unknown synthesizers"""

    def __init__(self, k: int = 5, threshold: float = 0.5):
        super().__init__()
        self.k = k
        self.threshold = threshold

    def forward(self, query_features, prototype_features):
        """Detect if speech is from unknown synthesizer"""
        # Compute distances
        distances = torch.cdist(query_features, prototype_features)
        knn_distances, _ = torch.topk(distances, self.k, largest=False)
        mean_distance = knn_distances.mean(dim=1)

        # Decision boundary
        is_unknown = mean_distance > self.threshold
        confidence = 1.0 - torch.sigmoid(mean_distance)

        return {
            "is_unknown": is_unknown,
            "confidence": confidence,
            "mean_knn_distance": mean_distance,
        }
