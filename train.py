from model import ResNet
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.utils.prune as prune
from pruning import Pruner
# Init wandb
import wandb
wandb.init(
    project="trying-things-out",
    name="second attempt",
    notes="25 epochs, 0.75 sparsity")

TRAIN_BATCH_SIZE = 100

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

epochs = 25
model = ResNet(10, nblock_layers=5).to(device)

# Log metrics with wandb
wandb.watch(model)

criterion = nn.CrossEntropyLoss()
lr = 0.01
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, 0.5)
should_prune = True

# pruning stuff

# only prune weights not biases
# exclude batch norm layers


def get_targets(model):
    submodules = [submodule for name, submodule in model.named_modules()
                  if type(submodule) != nn.BatchNorm2d]
    targets = []
    for submodule in submodules:
        for name, _ in submodule.named_parameters(recurse=False):
            if "weight" in name:
                targets.append((submodule, name))
    return targets


targets = get_targets(model)


pruning_frequency = 100
training_steps = epochs * len(trainloader)
pruning_steps = training_steps * 0.5 // pruning_frequency
start_step = training_steps * 0.1 // pruning_frequency
pruner = Pruner(targets, pruning_steps, start_step=start_step,
                final_sparsity=0.75)


def train():
    """Train the model for one epoch."""

    running_loss = 0.0
    running_acc = 0.0
    model.train()
    for i, data in enumerate(trainloader):
        step = len(trainloader) * (epoch - 1) + i + 1
        inputs, labels = data[0].to(device), data[1].to(device)

        # gradients are additive otherwise
        optimiser.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimiser.step()

        if should_prune and i % pruning_frequency == pruning_frequency - 1:
            pruner.prune()
        wandb.log({
            "training loss": loss,
            "sparsity": pruner.current_sparsity
        },
            step=step)
        # print loss
        running_loss += loss.item()
        running_acc += accuracy(outputs, labels)
        if i != 0 and i % 50 == 49:
            average_loss = running_loss/50
            average_acc = 100*running_acc/50
            wandb.log({"training accuracy": average_acc}, step=step)
            print("-"*80)
            print(f"batch: {i + 1} | ",
                  f"avg train loss: {average_loss:.5f}", " | "
                  f"avg train accuracy: {average_acc:.3f}%")
            running_loss = 0.0
            running_acc = 0.0


def evaluate():
    """Evaluate model performance on validation dataset"""
    step = epoch*len(trainloader)
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
    wandb.log({
        "validation loss": valid_loss,
        "validation accuracy": average_acc
    },
        step=step)
    return valid_loss


def accuracy(outputs, labels):
    """Calculate the average accuracy across a batch"""

    # get index
    preds = outputs.argmax(axis=1)
    # find rate of matches
    acc = float(((preds == labels)).sum())/len(preds)
    return acc


if __name__ == "__main__":

    best_loss = None
    pruned_best_loss = None
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
                if not should_prune:
                    with open('model.pt', "wb") as file:
                        torch.save(model, file)
                        print("-"*80)
                        print("best model saved")
                else:
                    # we only want to save if it's done pruning to the desired
                    # sparsity
                    if pruner.done_pruning:
                        if not pruned_best_loss or val_loss < pruned_best_loss:
                            pruned_best_loss = val_loss
                            with open('model.pt', "wb") as file:
                                torch.save(model, file)
                                print("-"*80)
                                print("best pruned model saved")
            else:
                # decay learning rate if no improvement
                scheduler.step()
                lr = optimiser.param_groups

    except KeyboardInterrupt:
        print("-"*80, "\n")
        print("exiting early")

    # load the best model
    with open("model.pt", 'rb') as file:
        model = torch.load(file, map_location=device)

    # Finalise model pruning (i.e. setting parameters to 0)
    # Removes mask and fixes parameters to zero

    # Needs to re-register the nn.Module objects as this is
    # a new model has been loaded into memory
    targets = get_targets(model)

    if should_prune:
        for module, params in targets:
            # mask = dict(module.named_buffers())["weight_mask"]
            # print("mask zeros", (mask == 0).sum())

            # remove needs to be passed "weight", even though technically 
            # that parameter isn't there anymore
            prune.remove(module, "weight")
            module._apply(lambda x: x)
        # Re-register the pruned parameters correctly
        # https://github.com/pytorch/pytorch/issues/33618

        with open("model.pt", 'wb') as f:
            torch.save(model, f)
    # parameters = torch.cat([param.data.flatten()
    #                         for param in list(model.parameters())])

    parameters = model.initial[0].weight.data.view(-1)

    sparsity = float((parameters == 0.0).sum())/len(parameters)
    print(targets[:2])
    print(parameters)
    print(f"final sparsity: {sparsity} (inc. unpruned BN layers etc.)")
    print(f"reported by pruner: {pruner.current_sparsity}")
