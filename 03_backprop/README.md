# 03 — Backpropagation MLP from Scratch

**Grade: 110% + Extra Credit**

A full backpropagation network implemented from scratch in NumPy. No PyTorch, no Scikit-Learn — just the math.

## Architecture

```
Backprop(n, h, m)
  wih: (n+1, h)  — input-to-hidden weights + bias
  who: (h+1, m)  — hidden-to-output weights + bias
  activation: sigmoid (+ derivative for delta computation)
  update: batch gradient descent with learning rate η
```

The training loop:
1. Forward pass through hidden layer → sigmoid → forward pass through output layer → sigmoid
2. Compute output delta: `δ_O = (t - O) * σ'(Onet)`
3. Backpropagate to hidden: `δ_H = (δ_O · Wₒᵤₜᵀ) * σ'(Hnet)`
4. Accumulate `ΔWᵢₕ` and `ΔWₒ` across the batch, then update

## Three tasks

**Part 1 — XOR** trains a 2→3→1 net on XOR with 10,000 iterations. XOR is not linearly separable, but with a hidden layer the MLP learns it. Includes 3D error surface visualization across two weight dimensions.

**Part 2 — "2-be or not 2-be"** trains a 196→25→1 binary classifier to detect the digit "2" in 14×14 images. Reports false-positive and false-negative rates with a structured confusion matrix.

**Part 3 — Full digit classifier** trains a 196→250→10 multi-class recognizer across all ten digits. Evaluates with a 10×10 confusion matrix; includes a 3D bar chart visualization.

## Extra credit

- 3D error surface across weight space (`xor_error_surface.png`)
- RMS training error curves for all three networks
- 3D confusion matrix for the 10-class classifier (`confusion3d.png`)

## Stack

`Python` · `NumPy` · `Matplotlib` · `tqdm`

## Plots

| File | Description |
|------|-------------|
| `XOR_error.png` | RMS error over 10,000 epochs for XOR |
| `2-not-2_error.png` | RMS error curve for binary digit classifier |
| `digits_error.png` | RMS error for 10-class classifier |
| `confusion3d.png` | 3D confusion matrix (green = correct, red = misclassified) |
| `xor_error_surface.png` | Error landscape across two weight dimensions |
