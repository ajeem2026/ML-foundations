# 05 — Convolutional Neural Network (MNIST)

**Grade: 100%**

A ConvNet built in PyTorch trained on MNIST for handwritten digit classification. Extends the MLP from assignment 4 with two convolutional stages.

## Architecture

```
MNISTConvNet
├── conv1: Conv2d(1→32, 5×5, same) → ReLU → MaxPool2d(2)
├── conv2: Conv2d(32→64, 5×5, same) → ReLU → MaxPool2d(2)
└── fc1:   Flatten → Linear(7×7×64→1024) → Dropout(0.5) → Linear(1024→10)
```

Feature map sizes:
- Input: 28×28×1
- After conv1 + pool: 14×14×32
- After conv2 + pool: 7×7×64
- Flattened: 3,136 → FC → 10 logits

## Training

- Dataset: MNIST (60,000 training images)
- Loss: `CrossEntropyLoss`
- Optimizer: SGD (lr=1e-2)
- Epochs: 500
- Device: GPU if available (`cuda` / `cpu` fallback)

## Evaluation

`cnn_test.py` loads the saved checkpoint (`mnist_conv.pt`) and reports:
- Overall test accuracy
- Test loss
- Full 10×10 confusion matrix (per-class breakdown)

## Files

| File | Purpose |
|------|---------|
| `cnn.py` | `MNISTConvNet` model definition |
| `cnn_train.py` | Training loop; saves `mnist_conv.pt` |
| `cnn_test.py` | Evaluation + confusion matrix |
| `slp.py` / `mlp.py` | SLP and MLP baselines for comparison |

## Stack

`Python` · `PyTorch` · `torchvision` · `NumPy`

## To run

```bash
# Train (saves checkpoint)
python cnn_train.py

# Evaluate
python cnn_test.py
```
