from model import ResNet
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.utils.prune as prune
from prune import Pruner

TRAIN_BATCH_SIZE = 128

EVAL_BATCH_SIZE = 128

# images are 32x32
# flip randomly for hopefully added robustness
# upscale them and normalise them to [-1, 1]
# ID for lanczos filter is some integer

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.Resize(64, interpolation=PIL.Image.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# no flips for test and valid
test_transform = transforms.Compose([
    transforms.Resize(64, interpolation=PIL.Image.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True,
                                        transform=train_transform)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=TRAIN_BATCH_SIZE,
                                          shuffle=True, num_workers=8,
                                          drop_last=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)

valset, testset = torch.utils.data.random_split(testset, [5000, 5000])

valloader = torch.utils.data.DataLoader(valset, batch_size=EVAL_BATCH_SIZE,
                                        shuffle=True, num_workers=8,
                                        drop_last=True)


testloader = torch.utils.data.DataLoader(testset, batch_size=EVAL_BATCH_SIZE,
                                         shuffle=False, num_workers=8,
                                         drop_last=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# if cuda is available use it
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------
# training

epochs = 30
model = ResNet(10, nblock_layers=7).to(device)
criterion = nn.CrossEntropyLoss()
lr = 0.01
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, 0.5)
should_prune = True

# pruning stuff

# only prune weights not biases
pruning_targets = [(model, param) for param 
                   in dict(model.named_parameters()).keys()
                   if "weight" in param]

pruning_frequency = 100
training_steps = epochs * len(trainloader)
pruning_steps = training_steps * 0.7 // pruning_frequency
start_step = training_steps * 0.1 // pruning_frequency
pruner = Pruner(pruning_targets, pruning_steps, start_step=0, final_sparsity=0.6)


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

        if should_prune and i % pruning_frequency == pruning_frequency - 1:
            pruner.prune()

        # print loss
        running_loss += loss.item()
        running_acc += accuracy(outputs, labels)
        if i != 0 and i % 50 == 49:
            average_loss = running_loss/50
            average_acc = 100*running_acc/50
            print("-"*80)
            print(f"batch: {i + 1} | ",
                  f"avg train loss: {average_loss:.5f}", " | "
                  f"avg train accuracy: {average_acc:.3f}%")
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
    print("-"*80)
    print(f"avg val loss: {valid_loss:.5f} | ",
          f"avg val accuracy: {average_acc:.3f}%")
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
            print("="*80) 
            print(f"epoch {epoch}")
            for params in optimiser.param_groups:
                print(f"learning rate: {params['lr']}", sep=" | ")
            train()
            val_loss = evaluate()

            # save the best model
            if not best_loss or val_loss < best_loss:
                best_loss = val_loss
                with open('model.pt', "wb") as file:
                    torch.save(model, file)
            else:
                # decay learning rate if no improvement
                scheduler.step()
                lr = optimiser.param_groups

    except KeyboardInterrupt:
        print("-"*80)
        print("exiting early")

    with open("model.pt", 'rb') as file:
        model = torch.load(file, map_location=device)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.

    # Finalise model pruning (i.e. setting parameters to 0)
    if should_prune:
        for module, params in pruning_targets:
            prune.remove(module, params)
        # Re-register the pruned parameters correctly
        # https://github.com/pytorch/pytorch/issues/33618
        model._apply(lambda x: x)
        with open("model.pt", 'wb') as f:
            torch.save(model, f)