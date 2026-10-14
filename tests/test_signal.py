"""Unit tests for SIGNAL"""

import unittest
import torch
from signal.signal import SIGNAL

class TestSIGNAL(unittest.TestCase):
    def setUp(self):
        self.model = SIGNAL(embedding_dim=1024, num_generators=10)

    def test_forward(self):
        embeddings = torch.randn(4, 1024)
        output = self.model(embeddings)
        self.assertIn('attribution_logits', output)
        self.assertIn('confidence_scores', output)
        self.assertEqual(output['attribution_logits'].shape[0], 4)

if __name__ == '__main__':
    unittest.main()
