# 01 — Linear Regression & Classification

**Grade: 110%**

From-scratch linear regression and binary classification using NumPy. No ML libraries — everything derived from the normal equations and least squares.

## What it does

`regression.py` walks through five parts:

1. **Univariate regression** — computes slope and intercept for y vs. x1, then y vs. x2 using the closed-form solution
2. **Multivariate regression** — fits a full linear model (w1·x1 + w2·x2 + b) via `np.linalg.lstsq`
3. **Binary classifier** — uses the regression plane to classify a binary label z; reports percentage correct
4. **Train/test split** — trains on {25, 50, 75} examples, tests on the remainder; compares against a zero-model baseline
5. **3D visualization** — plots scatter by class label + decision plane in 3D with Matplotlib

## Stack

`Python` · `NumPy` · `Matplotlib`

## Key result

A classifier derived purely from a regression plane correctly classifies held-out examples, with performance degrading predictably as training set size shrinks. Zero-model baseline confirms the classifier adds real signal.
