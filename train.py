from model import ResNet
import torch
import torchvision
import torchvision.transforms as transforms
import PIL
import matplotlib.pyplot as plt
import numpy as np

# images are 32x32
# flip randomly for hopefully added robustness
# upscale them and normalise them to [-1, 1]
# ID for lanczos filter is some integer

transform = transforms.Compose([
     transforms.RandomHorizontalFlip(p=0.1),
     transforms.RandomVerticalFlip(p=0.1),
     transforms.Resize(64, interpolation=PIL.Image.LANCZOS),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainset, valset = torch.utils.data.random_split(dataset, [40000, 10000])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=20,
                                          shuffle=True, num_workers=8)
valloader = torch.utils.data.DataLoader(valset, batch_size=20,
                                          shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=20,
                                         shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# if cuda is available use it
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# functions to show an image




