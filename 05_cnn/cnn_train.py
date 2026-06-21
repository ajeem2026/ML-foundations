import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import optim


from cnn import MNISTConvNet
import time 

# For reproducability
torch.manual_seed(0)

# Full Description of the Convolutional Layer

layer = nn.Conv2d(in_channels = 3,
                  out_channels = 64,
                  kernel_size = (5, 5),
                  stride = 2,
                  padding = 1
                  )


trainset = MNIST('.', train=True, download=True, 
                      transform=ToTensor())
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)


# # loading test dataset
# testset = MNIST(".", train=False, download=True, transform=ToTensor())
# testloader = DataLoader(testset, batch_size=64, shuffle=False)


lr = 1e-2
num_epochs = 500 #40

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MNISTConvNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)
 
if __name__ == '__main__':
 
    print('Training on device: {}'.format(device))
 
    start = time.perf_counter()
 
    for epoch in range(num_epochs):
        running_loss = 0.0
        num_correct = 0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = loss_fn(outputs, labels.to(device))
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
            _, idx = outputs.max(dim=1)
            num_correct += (idx == labels.to(device)).sum().item()
        print('Epoch: {} Loss: {:.4f} Accuracy: {:.4f}'.format(
            epoch + 1,
            running_loss / len(trainloader),
            num_correct / len(trainloader)))
 
    end = time.perf_counter()
    print('Training complete in: {:.0f}m {:.0f}s'.format(
        (end - start) // 60, (end - start) % 60))
 
    torch.save(model.state_dict(), 'mnist_conv.pt')
    print('Model saved to mnist_conv.pt')