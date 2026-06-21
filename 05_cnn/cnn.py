import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import optim

# # For reproducability
# torch.manual_seed(0)

# # Full Description of the Convolutional Layer

# layer = nn.Conv2d(in_channels = 3,
#                   out_channels = 64,
#                   kernel_size = (5, 5),
#                   stride = 2,
#                   padding = 1
#                   )


#Closing the Loop on MNIST with Convolutional Networks

class MNISTConvNet(nn.Module):
  def __init__(self):
    super(MNISTConvNet, self).__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(1, 32, 5, padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(32, 64, 5, padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    self.fc1 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(7*7*64, 1024),
        nn.Dropout(0.5),
        nn.Linear(1024, 10)
    )

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    return self.fc1(x)

# trainset = MNIST('.', train=True, download=True, 
#                       transform=ToTensor())
# trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# lr = 1e-4
# num_epochs = 40

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = MNISTConvNet().to(device)
# loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=lr)

# for epochs in range(num_epochs):
#   running_loss = 0.0
#   num_correct = 0
#   for inputs, labels in trainloader:
#     optimizer.zero_grad()
#     outputs = model(inputs.to(device))
#     loss = loss_fn(outputs, labels.to(device))
#     loss.backward()
#     running_loss += loss.item()
#     optimizer.step()
#     _, idx = outputs.max(dim=1)
#     num_correct += (idx == labels.to(device)).sum().item()
#   print('Loss: {} Accuracy: {}'.format(running_loss/len(trainloader),
#         num_correct/len(trainloader)))
  
# # Image Preprocessing Pipelines Enable More Robust Models

# from torchvision import transforms

# transform = transforms.Normalize(mean = (0.1307,),
#                                  std = (0.3081,)
#                                  )

# transform = transforms.Compose([
#       transforms.RandomCrop(224),
#       transforms.RandomHorizontalFlip(),
#       transforms.ColorJitter(brightness=0,
#                              contrast=0,
#                              saturation=0,
#                              hue=0),
#       transforms.ToTensor(),
#       transforms.Normalize(mean = (0.1307,),
#                            std = (0.3081,)
#                            )
#       ])

# # Accelerating Training with Batch Normalization

# layer = nn.BatchNorm2d(num_features=32,
#                        eps=1e-05,
#                        momentum=0.1,
#                        affine = True,
#                        track_running_stats = True)
     
# layer = nn.BatchNorm1d(num_features=32)

# # Group normalization for memory constrained learning tasks

# layer = nn.GroupNorm(num_groups=1,
#                      num_channels=32)

# # Building a Convolutional Network for CIFAR-10

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.block1 = nn.Sequential(
#             nn.Conv2d(1, 32, 3, 1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 64, 3, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#             nn.Dropout(0.25),
#         )
#         self.block2 = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(9216, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(128,10),
#             nn.BatchNorm1d(10)
#         )

#     def forward(self, x):
#         x = self.block1(x)
#         return self.block2(x)
    
# # Building a residual network with superhuman vision

