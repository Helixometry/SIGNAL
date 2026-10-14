# SIGNAL Architecture

## Overview

SIGNAL uses graph-augmented instance learning to solve two linked problems:
1. Generator Attribution - Identifying which TTS system produced synthetic speech
2. Open-Set Detection - Flagging unknown/novel synthesizers

## Key Components

### 1. Speech Foundation Models
- Mamba-based embeddings (best performance)
- XLSR embeddings
- Other speech foundation models
- Output: Fixed-dimensional embeddings (1024-dim typical)

### 2. Query-Conditioned Graph Construction
- Generator prototypes (learned parameters)
- Dynamic graph based on input speech embedding
- Models relationships among candidate generators

### 3. Graph Neural Network
- Node features: Generator prototypes
- Query: Input speech embedding
- Output: Generator-aware features for classification

### 4. Dual-Branch Inference

**Attribution Branch (GNN)**
- Closed-set generator classification
- Softmax over known generators
- Confidence per generator

**Open-Set Branch (KNN)**
- K-nearest neighbors in embedding space
- Distance-based confidence thresholding
- Rejects unknown synthesizers

## Results on DiffSSD

- **Known Generator Attribution**: 94.8% accuracy (Mamba embeddings)
- **Open-Set Detection**: 0.892 AUROC
- **Generalization to SingFake**: Strong transfer performance

## Performance Metrics

- Accuracy: Correct generator identification rate
- AUROC: Open-set detection area under ROC curve
- F1-Score: Harmonic mean of precision and recall
- KNN Confidence: Distance-based uncertainty estimation

## Advantages

✅ Unified attribution + open-set detection
✅ Graph-based reasoning about generator relationships
✅ Strong generalization to unseen synthesizers
✅ Efficient KNN-based open-set branch
✅ Works with multiple foundation models
