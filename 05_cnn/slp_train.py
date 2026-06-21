import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from slp import SLPClassifier
import time

torch.manual_seed(0)

train_dataset = MNIST(".", train=True, download=True, transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

in_dim, out_dim = 784, 10
lr = 1e-2
epochs = 500
loss_fn = nn.CrossEntropyLoss()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
classifier = SLPClassifier(in_dim, out_dim).to(device)
optimizer = optim.SGD(classifier.parameters(), lr=lr)

def train(classifier=classifier, optimizer=optimizer, epochs=epochs, loss_fn=loss_fn):
    classifier.train()
    start = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        for minibatch in train_loader:
            data, target = minibatch
            data = data.flatten(start_dim=1).to(device)
            target = target.to(device)
            out = classifier(data)
            computed_loss = loss_fn(out, target)
            computed_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += computed_loss.item()
        print("Epoch: {} train loss: {}".format(epoch + 1, running_loss / len(train_loader)))
    elapsed = time.time() - start
    print('Training on device: {}'.format(device))
    print(f"Training time: {elapsed:.2f}s")
    print("Saving network to slp.pt")
    torch.save(classifier.state_dict(), 'slp.pt')

train()