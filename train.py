from model import ResNet
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import PIL
import matplotlib.pyplot as plt
import numpy as np

TRAIN_BATCH_SIZE = 20

EVAL_BATCH_SIZE = 100

# images are 32x32
# flip randomly for hopefully added robustness
# upscale them and normalise them to [-1, 1]
# ID for lanczos filter is some integer

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.RandomVerticalFlip(p=0.25),
    transforms.Resize(64, interpolation=PIL.Image.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)
trainset, valset = torch.utils.data.random_split(dataset, [40000, 10000])
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=TRAIN_BATCH_SIZE,
                                          shuffle=True, num_workers=8,
                                          drop_last=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=EVAL_BATCH_SIZE,
                                        shuffle=True, num_workers=8,
                                        drop_last=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=EVAL_BATCH_SIZE,
                                         shuffle=False, num_workers=8,
                                         drop_last=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# if cuda is available use it
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------
# training

epochs = 20
model = ResNet(10).to(device)
criterion = nn.CrossEntropyLoss()
lr = 0.01
optimiser = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, 0.25)


def train():
    """Train the model for one epoch."""

    running_loss = 0.0
    running_acc = 0.0
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
        running_acc += accuracy(outputs, labels)
        if i != 0 and i % 200 == 0:
            average_loss = running_loss/200
            average_acc = 100*running_acc/200
            print("-"*20)
            print((f"batch: {i} | average training loss "
                   f"over last 200: {average_loss:.5f}"))
            print((f"batch: {i} | average training accuracy "
                   f"over last 200: {average_acc:.3f}%"))
            running_loss = 0.0
            running_acc = 0.0


def evaluate():
    """Evaluate model performance on validation dataset"""
    valid_loss = 0.0
    running_acc = 0.0
    model.eval()
    with torch.no_grad():
        for data in valloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # loss is averaged across each batch
            valid_loss += loss
            running_acc += accuracy(outputs, labels)
    valid_loss = valid_loss/len(valloader)
    average_acc = running_acc/len(valloader) * 100
    print("-"*20)
    print(f"average validation loss: {valid_loss:.5f}")
    print(f"average validation accuracy: {average_acc:.3f}%")
    return valid_loss


def accuracy(outputs, labels):
    """Calculate the average accuracy across a batch"""

    # get index
    preds = outputs.argmax(axis=1)
    # find rate of matches
    acc = float(((preds == labels)).sum())/len(preds)
    return acc


if __name__ == "__main__":
    with open('model.pt', "wb") as file:
        torch.save(model, file)
    best_loss = None
    try:
        for epoch in range(1, epochs+1):
            print(f"epoch {epoch}")
            print()
            train()
            val_loss = evaluate()

            # save the best model 
            if not best_loss or val_loss <  best_loss:
                best_loss = val_loss
                with open('model.pt', "wb") as file:
                    torch.save(model, file)
            else:
                # decay learning rate if no improvement
                scheduler.step()
                lr = optimiser.param_groups

    except KeyboardInterrupt:
        print("-"*20)
        print("exiting early")

