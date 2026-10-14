"""Speech foundation models for embedding extraction"""

import torch.nn as nn


class MambaEmbedding(nn.Module):
    """Mamba-based speech embedding extractor"""

    def __init__(self, output_dim: int = 1024):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, audio):
        """Extract Mamba embeddings"""
        return audio


class XLSREmbedding(nn.Module):
    """XLSR-based speech embedding extractor"""

    def __init__(self, output_dim: int = 1024):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, audio):
        """Extract XLSR embeddings"""
        return audio
