# ml-foundations

Neural network implementations built from scratch and with PyTorch — coursework from W&L's AI curriculum (Physics/CS, GPA 3.9). Five progressive assignments covering linear models through CNNs, with extra credit on three of five.

**Stack:** Python · NumPy · PyTorch · Matplotlib · Scikit-Learn

---

## Sub-Projects

| # | Topic | Key Concepts | 
|---|-------|-------------|
| [01](./01_regression/) | Linear Regression & Classification | Least squares, train/test split, 3D decision boundary | 
| [02](./02_perceptron/) | Perceptron from Scratch | Delta rule, digit recognition, FP/FN analysis | 
| [03](./03_backprop/) | Backpropagation MLP from Scratch | Sigmoid, batch updates, 10-class digit classifier, error surface | 
| [04](./04_pytorch_mlp/) | PyTorch MLP | nn.Module, CrossEntropyLoss, SGD, MNIST | 
| [05](./05_cnn/) | Convolutional Neural Network | Conv2d, MaxPool, Dropout, MNIST, confusion matrix | 

---

## Highlights

**Backprop from scratch** (`03_backprop/backprop.py`) — No ML frameworks. Full forward pass, output delta, hidden delta, batch weight accumulation. Trains a 196→250→10 digit recognizer and produces a 3D error surface over weight space.

**CNN in PyTorch** (`05_cnn/cnn.py`) — `MNISTConvNet`: Conv2d(1→32→64) + MaxPool2d + Dropout(0.5) + FC(1024→10). Trained 500 epochs with SGD. Evaluated with a 10×10 confusion matrix.

**XOR failure analysis** (`02_perceptron/`) — Demonstrates why a single-layer perceptron cannot solve XOR, then fixes it in assignment 3 with a hidden layer.
