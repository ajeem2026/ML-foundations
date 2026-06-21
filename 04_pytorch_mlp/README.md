# 04 — PyTorch MLP (SLP + MLP)

Transition from NumPy-from-scratch to PyTorch's `nn.Module` API. Implements a Single-Layer Perceptron (SLP) and Multi-Layer Perceptron (MLP) on MNIST.

## Models

**SLPClassifier** (`slp.py`)
```python
nn.Sequential(nn.Linear(in_dim, out_dim, bias=True))
```

**BaseClassifier** (`mlp.py`)
```python
nn.Sequential(
    nn.Linear(in_dim, feature_dim),
    nn.ReLU(),
    nn.Linear(feature_dim, out_dim)
)
```

## Training pipeline

- Dataset: MNIST (60K train / 10K test) via `torchvision`
- Loss: `CrossEntropyLoss`
- Optimizer: `SGD` (lr=1e-3)
- DataLoader with minibatch shuffle

## Files

| File | Purpose |
|------|---------|
| `slp.py` | SLP model definition |
| `mlp.py` | MLP model definition + PyTorch tensor reference notes |
| `slp_train.py` | SLP training loop |
| `slp_test.py` | SLP evaluation |
| `mlp_train.py` | MLP training loop |
| `mlp_test.py` | MLP evaluation |

## Stack

`Python` · `PyTorch` · `torchvision` · `Matplotlib`
