import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from mlp import BaseClassifier
import numpy as np


# loading test dataset
test_dataset = MNIST(".", train=False, download=True, transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# more loading and extracting dimensions from the saved state dict
state_dict = torch.load('mnist_mlp.pt', weights_only=True)
in_dim = state_dict['classifier.0.weight'].shape[1]
feature_dim = state_dict['classifier.0.weight'].shape[0]
out_dim = state_dict['classifier.2.weight'].shape[0]

#print(f"Loaded network dimensions: in={in_dim}, hidden={feature_dim}, out={out_dim}")


classifier = BaseClassifier(in_dim, feature_dim, out_dim)
classifier.load_state_dict(state_dict)

loss_fn = nn.CrossEntropyLoss()

def test(classifier=classifier, loss_fn=loss_fn):
    classifier.eval()
    accuracy = 0.0
    computed_loss = 0.0
    
    prediction=[]
    targets=[]

    with torch.no_grad():
        for data, target in test_loader:
            data = data.flatten(start_dim=1)
            out = classifier(data)
            _, preds = out.max(dim=1)
            computed_loss += loss_fn(out, target)
            accuracy += torch.sum(preds == target)
            
            prediction.extend(preds.numpy())
            targets.extend(target.numpy())
            

    print("Test loss: {}, test accuracy: {}".format(
        computed_loss.item() / (len(test_loader) * 64),
        accuracy * 100.0 / (len(test_loader) * 64)))
    
    conf = np.zeros((10, 10), dtype=int)
    for actual, pred in zip(targets, prediction):
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
