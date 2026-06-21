import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from mlp import BaseClassifier

import time


torch.manual_seed(0)

train_dataset = MNIST(".", train=True, download=True, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

in_dim, feature_dim, out_dim = 784, 256, 10
lr = 1e-1
epochs = 40
particular_loss_value=0.37

# with 20 epochs: test accurary was 88.99%
# with 40 epochs: test accuracy was 90.66%
# with 60 epochs: test accuracy was 91.52%

# so more epochs == better accuracy, but with diminishing returns. 60 epochs is a good balance between training time and accuracy.
loss_fn = nn.CrossEntropyLoss()

classifier = BaseClassifier(in_dim, feature_dim, out_dim)
optimizer = optim.SGD(classifier.parameters(), lr=lr)

# def train(classifier=classifier, optimizer=optimizer, epochs=epochs, loss_fn=loss_fn):
#     classifier.train()
#     start = time.time()
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for minibatch in train_loader:
#             data, target = minibatch
#             data = data.flatten(start_dim=1)
#             out = classifier(data)
#             computed_loss = loss_fn(out, target)
#             computed_loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             running_loss += computed_loss.item()
#         print("Epoch: {} train loss: {}".format(epoch+1, running_loss/len(train_loader)))
#     elapsed = time.time() - start
#     print(f"Training time: {elapsed:.2f}s")
#     print("Saving network to mnist.pt")
#     torch.save(classifier.state_dict(), 'mnist.pt')

def train(classifier=classifier, optimizer=optimizer, loss_fn=loss_fn):
    classifier.train()
    start = time.time()
    epoch = 0
    while True:
        running_loss = 0.0
        epoch += 1
        for minibatch in train_loader:
            data, target = minibatch
            data = data.flatten(start_dim=1)
            out = classifier(data)
            computed_loss = loss_fn(out, target)
            computed_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += computed_loss.item()
        avg_loss = running_loss / len(train_loader)
        print("Epoch: {} train loss: {}".format(epoch, avg_loss))
        if avg_loss < particular_loss_value:
            print(f"Reached target loss {avg_loss:.6f} after {epoch} epochs")
            break
    elapsed = time.time() - start
    print(f"Training time: {elapsed:.2f}s")
    print("Saving network to mnist.pt")
    torch.save(classifier.state_dict(), 'mnist.pt')

train()





