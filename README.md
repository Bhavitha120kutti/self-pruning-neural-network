# Self-Pruning Neural Network

## Overview
This project implements a neural network that dynamically prunes its own weights during training using learnable gates and L1 regularization.

## Key Idea
Each weight is multiplied by a learnable gate (0 to 1).  
If gate → 0, the weight is effectively removed.

## Formula
Loss = CrossEntropy + λ × Sparsity Loss

## Dataset
- CIFAR-10

## Results
| Lambda | Accuracy | Sparsity |
|--------|---------|---------|
| 1e-5   | XX%     | XX%     |
| 1e-4   | XX%     | XX%     |
| 1e-3   | XX%     | XX%     |

## How to Run
```bash
pip install torch torchvision matplotlib
python pruning_model.py
