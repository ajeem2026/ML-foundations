import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from cnn import MNISTConvNet
import numpy as np


# loading test dataset
testset = MNIST(".", train=False, download=True, transform=ToTensor())
testloader = DataLoader(testset, batch_size=64, shuffle=False)


#print(f"Loaded network dimensions: in={in_dim}, hidden={feature_dim}, out={out_dim}")
#=============
#change all classifier to model AND BASECLASSIFIER TO MNISTCONVNET
#=============
#test : whej you face foraward issue, just comment out hte flatten bc nn's want the 4d tensor, not the 2d one.


state_dict = torch.load('mnist_conv.pt', weights_only=True)
model = MNISTConvNet()
model.load_state_dict(state_dict)
 
loss_fn = nn.CrossEntropyLoss()

def test(model=model, loss_fn=loss_fn):
    model.eval()
    accuracy = 0.0
    computed_loss = 0.0
 
    predictions = []
    targets = []
 
    with torch.no_grad():
        for data, target in testloader:
            # No flatten needed — CNN expects the full 4D tensor (batch, channel, H, W)
            out = model(data)
            _, preds = out.max(dim=1)
            computed_loss += loss_fn(out, target)
            accuracy += torch.sum(preds == target)
 
            predictions.extend(preds.numpy())
            targets.extend(target.numpy())
 
    print("Test loss: {}, test accuracy: {}".format(
        computed_loss.item() / (len(testloader) * 64),
        accuracy * 100.0 / (len(testloader) * 64)))
 
    conf = np.zeros((10, 10), dtype=int)
    for actual, pred in zip(targets, predictions):
        conf[actual][pred] += 1
 
    print("\nConfusion Matrix:\n")
    print(f"{'':>4}", end="")
    for i in range(10):
        print(f"{i:>5}", end="")
    print()
    print("-" * 54)
    for i in range(10):
        print(f"{i:>3} |", end="")
        for j in range(10):
            print(f"{conf[i][j]:>5}", end="")
        print()
 
test()
 
