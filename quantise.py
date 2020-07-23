import timeit
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import PIL

EVAL_BATCH_SIZE = 128

test_transform = transforms.Compose([
    transforms.Resize(64, interpolation=PIL.Image.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)

valset, testset = torch.utils.data.random_split(testset, [5000, 5000])

valloader = torch.utils.data.DataLoader(valset, batch_size=EVAL_BATCH_SIZE,
                                        shuffle=True, num_workers=8,
                                        drop_last=True)


def accuracy(outputs, labels):
    """Calculate the average accuracy across a batch"""

    # get index
    preds = outputs.argmax(axis=1)
    # find rate of matches
    acc = float(((preds == labels)).sum())/len(preds)
    return acc


def evaluate(model, dataloader, criterion, device="cpu"):
    """Evaluate model performance on validation dataset"""
    valid_loss = 0.0
    running_acc = 0.0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # loss is averaged across each batch
            valid_loss += loss
            running_acc += accuracy(outputs, labels)
    valid_loss = valid_loss/len(dataloader)
    average_acc = running_acc/len(dataloader) * 100
    print("-"*80)
    print(f"avg val loss: {valid_loss:.5f} | ",
          f"avg val accuracy: {average_acc:.3f}%")
    return valid_loss, average_acc


criterion = nn.CrossEntropyLoss()
ncal_batches = 10
float_model = torch.load("model.pt", map_location="cpu")
float_model.eval()

# fuse conv bn prelu
# float_model.fuse_model()

# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
float_model.qconfig = torch.quantization.default_qconfig
print(float_model.qconfig)

# insert observers
torch.quantization.prepare(float_model, inplace=True)

# Calibrate with validation set
evaluate(float_model, valloader, criterion)
# for n, p in float_model.named_buffers():
#    print(n, p)
# # convert to quantized model
# quant_model = torch.quantization.convert(float_model, inplace=False)


# evaluate(quant_model, valloader, criterion)
