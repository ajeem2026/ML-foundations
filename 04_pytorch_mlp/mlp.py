# # ================================PyTorch Tensors===============================
# import torch
# torch.manual_seed(0)

# #================================Tensor Init================================

# arr = [1,2]
# tensor = torch.tensor(arr)
# val = 2.0
# tensor = torch.tensor(val)

# import numpy as np
# np_arr = np.array([1,2])
# x_t = torch.from_numpy(np_arr)

# zeros_t = torch.zeros((2,3)) # Returns 2x3 tensor of zeros
# ones_t = torch.ones((2,3)) # Returns 2x3 tensor of ones
# rand_t = torch.randn((2,3)) # Returns 2x3 tensor of random numbers

# #================================Tensor Attributes================================

# print(zeros_t.shape) # Returns torch.Size([2, 3])

# x_t = torch.tensor(2.0)
# print(x_t.dtype) # Returns torch.float32

# arr = [1,2]
# x_t = torch.tensor(arr, dtype=torch.float32)

# print(x_t.device) # Returns device(type='cpu') by default

# # PyTorch will use GPU if it's available
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# arr = [1,2]
# x_t = torch.tensor(arr, dtype=torch.float32, device=device)

# x_t = x_t.to(device, dtype=torch.int)

# #================================Tensor Operations================================
# c = 10
# x_t = x_t*c

# x1_t = torch.zeros((1,2))
# x2_t = torch.ones((1,2))
# x1_t + x2_t
# # returns tensor([[1., 1.]])

# x1_t = torch.tensor([[1,2],[3,4]])
# x2_t = torch.tensor([[1,2,3],[4,5,6]])
# torch.matmul(x1_t, x2_t) # Returns tensor([[9,12,15],[19,26,33]])

# i,j,k = 0,1,1
# x3_t = torch.tensor([[[3,7,9],[2,4,5]],[[8,6,2],[3,9,1]]])
# print(x3_t)
# # out:
# # tensor([[[3, 7, 9],
# #          [2, 4, 5]],
# #         [[8, 6, 2],
# #          [3, 9, 1]]])

# x3_t[i,j,k]
# # out:
# # tensor(4)


# x3_t[0] # Returns the matrix at position 0 in tensor
# x3_t[0,:,:] # Also returns the matrix at position 0 in tensor!
# # out:
# # tensor([[3, 7, 9],
# #         [2, 4, 1]])
     
# x3_t[0,1:3,:]
# # returns tensor([[2, 4, 5]])

# x3_t[0,1,2] = 1
# # out:
# # tensor([[[3, 7, 9],
# #          [2, 4, 1]],

# #         [[8, 6, 2],
# #          [3, 9, 1]]])

# x_t = torch.randn(2,3,4)
# sub_tensor = torch.randn(2,4)
# x_t[0,1:3,:] = sub_tensor


# x_t[0,1:3,:] = 1
# sub_tensor = torch.randn(1,4)
# x_t[0,1:3,:] = sub_tensor

# #================================Gradients in PyTorch================================

# x = torch.tensor(2.0, requires_grad=True)
# y = torch.tensor(3.0, requires_grad=True)
# z = torch.tensor(1.5, requires_grad=True)
# f = x**2+y**2+z**2
# f.backward()
# x.grad, y.grad, z.grad
# # out:
# # (tensor(4.), tensor(6.), tensor(3.))

# #================================The PyTorch nn module================================

import torch.nn as nn

# in_dim, out_dim = 256, 10
# vec = torch.randn(256)
# layer = nn.Linear(in_dim, out_dim, bias=True)
# out = layer(vec)

# W = torch.rand(10,256)
# b = torch.zeros(10,1)
# out = torch.matmul(W, vec) + b

# in_dim, feature_dim, out_dim = 784, 256, 10
# vec = torch.randn(784)
# layer1 = nn.Linear(in_dim, feature_dim, bias=True)
# layer2 = nn.Linear(feature_dim, out_dim, bias=True)
# out = layer2(layer1(vec))


# relu = nn.ReLU()
# out = layer2(relu(layer1(vec)))


# class BaseClassifier(nn.Module):
#   def __init__(self, in_dim, feature_dim, out_dim):
#     super(BaseClassifier, self).__init__()
#     self.layer1 = nn.Linear(in_dim, feature_dim, bias=True)
#     self.layer2 = nn.Linear(feature_dim, out_dim, bias=True)
#     self.relu = nn.ReLU()

#   def forward(self, x):
#     x = self.layer1(x)
#     x = self.relu(x)
#     out = self.layer2(x)
#     return out


# no_examples = 10
# in_dim, feature_dim, out_dim = 784, 256, 10
# x = torch.randn((no_examples, in_dim))
# classifier = BaseClassifier(in_dim, feature_dim, out_dim)
# out = classifier(x)

# loss = nn.CrossEntropyLoss()
# target = torch.tensor([0,3,2,8,2,9,3,7,1,6])
# computed_loss = loss(out, target)
# computed_loss.backward()


# for p in classifier.parameters():
#   print(p.shape)

# # out:
# # torch.Size([256, 784])
# # torch.Size([256])
# # torch.Size([10, 256])
# # torch.Size([10])

# from torch import optim

# lr = 1e-3
# optimizer = optim.SGD(classifier.parameters(), lr=lr)

# optimizer.step() # Updates parameters via SGD
# optimizer.zero_grad() # Zeroes out gradients between minibatches

# #================================PyTorch Datasets and Dataloaders================================

# from torch.utils.data import Dataset, DataLoader

# import numpy as np
# labels = np.array([2, 0, 4, 1])
# np.save('labels',labels)

# labels_1 = np.load('labels.npy')
# print(labels_1)


# #================================First download files for use in custom ImageDataset example================================

# # donezo

# import os
# from PIL import Image
# from torchvision import transforms

# class ImageDataset(Dataset):
#   def __init__(self, img_dir, label_file):
#     super(ImageDataset, self).__init__()
#     self.img_dir = img_dir
#     self.labels = torch.tensor(np.load(label_file, allow_pickle=True))
#     self.transforms = transforms.ToTensor()
  
#   def __getitem__(self, idx):
#     img_pth = os.path.join(self.img_dir, "img_{}.jpg".format(idx))
#     img = Image.open(img_pth)
#     img = self.transforms(img).flatten()
#     label = self.labels[idx]
#     return {"data":img, "label":label}
  
#   def __len__(self):
#     return len(self.labels)


# train_dataset = ImageDataset(img_dir='./data/train/',
#                              label_file='./data/train/labels.npy')

# train_loader = DataLoader(train_dataset, 
#                           batch_size=4, 
#                           shuffle=True)


# for minibatch in train_loader:
#   data, labels = minibatch['data'], minibatch['label']
#   print(data)
#   print(labels)
  
  
# #================================Building the MNIST Classifer in PyTorch================================

# import matplotlib.pyplot as plt
# import torch
# from torch import optim
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision.datasets import MNIST
# from torchvision.transforms import ToTensor

# # For reproducability
# print(torch.manual_seed(0))


# class BaseClassifier(nn.Module):
#   def __init__(self, in_dim, feature_dim, out_dim):
#     super(BaseClassifier, self).__init__()
#     self.classifier = nn.Sequential(
#         nn.Linear(in_dim, feature_dim, bias=True),
#         nn.ReLU(),
#         nn.Linear(feature_dim, out_dim, bias=True)
#     )
    
#   def forward(self, x):
#     return self.classifier(x)
    

# # Load in MNIST dataset from PyTorch
# train_dataset = MNIST(".", train=True, 
#                       download=True, transform=ToTensor())
# test_dataset = MNIST(".", train=False, 
#                      download=True, transform=ToTensor())
# train_loader = DataLoader(train_dataset, 
#                           batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, 
#                          batch_size=64, shuffle=False)

# # Instantiate model, optimizer, and hyperparameter(s)
# in_dim, feature_dim, out_dim = 784, 256, 10
# lr=1e-3
# loss_fn = nn.CrossEntropyLoss()
# epochs=40
# classifier = BaseClassifier(in_dim, feature_dim, out_dim)
# optimizer = optim.SGD(classifier.parameters(), lr=lr)

# def train(classifier=classifier,
#           optimizer=optimizer,
#           epochs=epochs,
#           loss_fn=loss_fn):

#   classifier.train()
#   loss_lt = []
#   for epoch in range(epochs):
#     running_loss = 0.0
#     for minibatch in train_loader:
#       data, target = minibatch
#       data = data.flatten(start_dim=1)
#       out = classifier(data)
#       computed_loss = loss_fn(out, target)
#       computed_loss.backward()
#       optimizer.step()
#       optimizer.zero_grad()
#       # Keep track of sum of loss of each minibatch
#       running_loss += computed_loss.item()
#     loss_lt.append(running_loss/len(train_loader))
#     print("Epoch: {} train loss: {}".format(epoch+1, running_loss/len(train_loader)))

#   plt.plot([i for i in range(1,epochs+1)], loss_lt)
#   plt.xlabel("Epoch")
#   plt.ylabel("Training Loss")
#   plt.title(
#       "MNIST Training Loss: optimizer {}, lr {}".format("SGD", lr))
#   plt.show()

#   # Save state to file as checkpoint
#   torch.save(classifier.state_dict(), 'mnist.pt')
  
# def test(classifier=classifier, 
#           loss_fn = loss_fn):
#     classifier.eval()
#     accuracy = 0.0
#     computed_loss = 0.0

#     with torch.no_grad():
#         for data, target in test_loader:
#             data = data.flatten(start_dim=1)
#             out = classifier(data)
#             _, preds = out.max(dim=1)

#             # Get loss and accuracy
#             computed_loss += loss_fn(out, target)
#             accuracy += torch.sum(preds==target)
            
#         print("Test loss: {}, test accuracy: {}".format(
#             computed_loss.item()/(len(test_loader)*64), accuracy*100.0/(len(test_loader)*64)))


# train()
# test()


class BaseClassifier(nn.Module):
    def __init__(self, in_dim, feature_dim, out_dim):
        super(BaseClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, feature_dim, bias=True),
            nn.ReLU(),
            nn.Linear(feature_dim, out_dim, bias=True)
        )

    def forward(self, x):
        return self.classifier(x)