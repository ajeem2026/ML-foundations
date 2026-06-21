# 02 — Perceptron from Scratch

**Grade: 100%**

A Perceptron classifier implemented from scratch in NumPy using the delta rule. No ML frameworks.

## What it does

`perceptron.py` covers three parts:

1. **Boolean functions** — trains a perceptron to learn AND and OR. Demonstrates that XOR is not linearly separable (perceptron fails no matter how many iterations you give it)
2. **Digit visualization** — loads 2,500 14×14 handwritten digit patterns from a flat text file; renders each digit as ASCII art in the terminal
3. **Binary digit classifier** — trains a perceptron to distinguish digit "2" from all others on the full 2,500-pattern training set; evaluates on a separate test set with false-positive and false-negative rates

## Architecture

```
Perceptron(n, m)
  wih: (n+1, m)  — weights + bias column
  train(inputs, targets, iters)  — batch delta rule
  test(inputs)  — forward pass with step activation
```

## Stack

`Python` · `NumPy`

## Key result

XOR fails (expected — motivates the need for hidden layers introduced in assignment 3). The "2 vs. not-2" classifier demonstrates real-world applicability of the perceptron rule on structured image data.
