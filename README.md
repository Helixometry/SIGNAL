# SIGNAL: Graph-Augmented Instance Learning for Synthetic Speech Attribution and Open-Set Detection

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-EACL%202026-blue)](https://aclanthology.org/2026.eacl-long.250/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)](https://pytorch.org/)

Forensic attribution and open-set detection of AI-synthesized speech using graph neural networks

</div>

## Overview

**SIGNAL** is a novel framework for two critical tasks in synthetic speech forensics:
1. **Generator Attribution** - Identifying which synthesizer produced a piece of synthetic speech
2. **Open-Set Detection** - Flagging speech from unknown synthesizers not seen during training

By combining speech foundation models with graph-based reasoning and open-set-aware inference, SIGNAL enables robust detection and attribution across known and unknown speech generators.

### Key Innovation

SIGNAL pairs foundation model embeddings with Graph Neural Networks (GNNs) to reason about relationships among generator prototypes, enabling:
- **Query-conditioned graphs** - Dynamic graph construction based on input speech
- **KNN-based open-set rejection** - Confidence thresholding for unknown synthesizers
- **Strong performance** - State-of-the-art results on DiffSSD and SingFake benchmarks

## Paper

**Bridging Attribution and Open-Set Detection using Graph-Augmented Instance Learning in Synthetic Speech**

- **Authors**: Mohd Mujtaba Akhtar, Girish, Farhan Sheth, Muskaan Singh
- **Conference**: EACL 2026 (Volume 1: Long Papers), Rabat, Morocco
- **Pages**: 5404–5413
- **Paper Link**: [ACL Anthology](https://aclanthology.org/2026.eacl-long.250/)

## Problem Statement

Current synthetic speech detection systems face two major limitations:
1. **Attribution only** - Identify known generators but assume closed-set scenario
2. **No open-set capability** - Fail to detect unknown or novel synthesizers
3. **Limited generalization** - Poor transfer to TTS systems not seen during training

SIGNAL addresses both attribution and open-set detection in a unified framework.

## SIGNAL Framework

### Architecture

```
Input Synthetic Speech Audio
    ↓
┌─────────────────────────────┐
│   Speech Foundation Model   │
│   Embedding Extraction      │
│   (Mamba/XLSR/etc.)        │
└──────────┬──────────────────┘
           ↓
┌─────────────────────────────┐
│   Query-Conditioned Graph   │
│   Construction              │
│   (Generator Prototypes)    │
└──────────┬──────────────────┘
           ↓
┌─────────────────────────────┐
│   Graph Neural Network      │
│   (Node Classification)     │
└──────────┬──────────────────┘
           ↓
    ┌──────┴──────┐
    ↓             ↓
┌────────┐   ┌───────────┐
│ GNN    │   │ KNN Open- │
│Output  │   │ Set       │
│(Closed)│   │ Detection │
└────┬───┘   └─────┬─────┘
     └──────┬──────┘
            ↓
    Final Prediction:
    - Known Generator ID
    - Or "Unknown"
```

### Key Components

1. **Foundation Models** - Extract speech embeddings (Mamba, XLSR, etc.)
2. **Graph Construction** - Build query-conditioned graphs over generator prototypes
3. **GNN Encoder** - Graph neural network for reasoning about generator relationships
4. **Dual-Branch Inference**:
   - GNN branch: Closed-set attribution
   - KNN branch: Open-set detection with confidence thresholding

## Datasets

### DiffSSD (Diffusion-based Synthetic Speech Dataset)
- Real speech samples
- Diffusion-based TTS from multiple commercial and open-source systems
- Multiple speakers and languages
- Multi-codec synthesis

### SingFake
- Singing voice synthesis evaluation benchmark
- Cross-domain generalization testing
- Known and unknown synthesizer evaluation

## Quick Start

### Installation

```bash
git clone https://github.com/Helixometry/SIGNAL.git
cd SIGNAL
pip install -r requirements.txt
```

### Training

```bash
python scripts/train.py --config configs/config_signal.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint models/signal_best.pt --test-data data/diffssd_test
```

### Inference

```bash
python scripts/inference.py --audio synthetic_speech.wav --checkpoint models/signal_best.pt
```

## Results

### Performance on DiffSSD

| Model | Known Generators (Accuracy) | Open-Set (AUROC) | F1-Score |
|-------|---------------------------|------------------|----------|
| Baseline | 87.3% | 0.762 | 0.851 |
| XLSR-based | 91.2% | 0.834 | 0.906 |
| **SIGNAL (Mamba)** | **94.8%** | **0.892** | **0.941** |

### Generalization to SingFake

- Strong transfer to out-of-distribution singing voice synthesis
- Robust open-set detection across unknown synthesizers
- Consistent improvements over single-model baselines

## Repository Structure

```
SIGNAL/
├── README.md
├── LICENSE
├── setup.py
├── requirements.txt
│
├── signal/                      # Main package
│   ├── __init__.py
│   ├── signal.py                # SIGNAL model
│   ├── graph_encoder.py         # Graph neural network
│   ├── foundation_models.py     # Speech embeddings (Mamba, XLSR)
│   ├── open_set_detector.py     # KNN-based open-set detection
│   ├── utils.py
│   └── metrics.py
│
├── data/                        # Data handling
│   ├── __init__.py
│   ├── loaders.py               # DiffSSD, SingFake loaders
│   ├── preprocessor.py          # Audio preprocessing
│   └── augmentation.py          # Data augmentation
│
├── scripts/                     # Training/evaluation
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── __init__.py
│
├── configs/                     # Configuration files
│   ├── config_signal.yaml
│   └── config_baseline.yaml
│
├── tests/                       # Unit tests
│   ├── test_signal.py
│   ├── test_graph_encoder.py
│   └── __init__.py
│
└── docs/                        # Documentation
    ├── ARCHITECTURE.md
    ├── TRAINING.md
    └── DATASETS.md
```

## Key Features

🔍 **Dual-Task Framework** - Attribution + open-set detection in one model  
📊 **Graph-Based Reasoning** - Query-conditioned GNNs for generator relationships  
🎯 **Open-Set Capable** - Confident rejection of unknown synthesizers  
🌍 **Strong Generalization** - Transfer to unseen TTS systems (SingFake)  
🎤 **Multiple Foundations** - Support for Mamba, XLSR, and other embeddings  
⚡ **Efficient Inference** - Fast KNN-based open-set detection  
📈 **SOTA Results** - 94.8% accuracy on known generators, 0.892 AUROC on open-set  

## Citation

```bibtex
@inproceedings{akhtar2026signal,
  title={Bridging Attribution and Open-Set Detection using Graph-Augmented Instance Learning in Synthetic Speech},
  author={Akhtar, Mohd Mujtaba and Girish and Sheth, Farhan and Singh, Muskaan},
  booktitle={Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2026)},
  pages={5404--5413},
  year={2026},
  address={Rabat, Morocco}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Authors

- **Mohd Mujtaba Akhtar**
- **Girish**
- **Farhan Sheth**
- **Muskaan Singh**

## Acknowledgments

- EACL 2026 organizers
- DiffSSD and SingFake dataset providers
- Speech foundation model researchers
- Graph neural network community

---

<div align="center">

**Advancing forensic attribution and open-set detection of AI-synthesized speech**

</div>
