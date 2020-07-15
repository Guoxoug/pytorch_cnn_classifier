from model import ResNet
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import PIL
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 20

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=8)
valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# if cuda is available use it
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#------------------------------------------------------------------------------
# training

epochs = 10
model = ResNet(10).to(device)
criterion = nn.CrossEntropyLoss()
lr=0.01
optimiser = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, 0.25)

def train():
    """Train the model for one epoch."""
    running_loss = 0.0
    model.train()
    for i, data in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)

        # gradients are additive otherwise 
        optimiser.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        # print loss
        running_loss += loss.item()
        if i != 0 and i % 200 == 0:
            average_loss = running_loss/(200*BATCH_SIZE)
            print(f"batch: {i} | average training loss: {average_loss:.5E}")
            running_loss = 0.0

def evaluate():
    """Evaluate model performance on validation dataset"""

    model.eval()
    
for epoch in range(1, epochs+1):
    print("epoch {epoch}")
    train()









