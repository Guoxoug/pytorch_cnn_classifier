from timeit import default_timer as timer
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import PIL
from model import ResNet
from quant_model import QuantResNet

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
                                        shuffle=False, num_workers=8,
                                        drop_last=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=EVAL_BATCH_SIZE,
                                        shuffle=False, num_workers=8,
                                        drop_last=True)                     


def accuracy(outputs, labels):
    """Calculate the average accuracy across a batch"""

    # get index
    preds = outputs.argmax(axis=1)
    # find rate of matches
    acc = float(((preds == labels)).sum())/len(preds)
    return acc


def evaluate(model, dataloader, criterion):
    """Evaluate model performance on validation dataset"""
    valid_loss = 0.0
    running_acc = 0.0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
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

# saved model does not have stubs
# save state dict and load into model with stubs
float_model = QuantResNet(10)
float_model.load_state_dict(torch.load("model_state_dict.pt")) 

# fuse conv bn relu]
print(float_model.res1.net[0])
# float_model.fuse_model()

# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
float_model.qconfig = torch.quantization.default_qconfig



# insert observers
quant_model = torch.quantization.prepare(float_model)
quant_model.eval()
# print(quant_model.res1.net[0])
# for n, p in float_model.named_buffers():
#    print(n, p)
# Calibrate with validation set
evaluate(quant_model, valloader, criterion)
# for _ in range(10):
#     images, labels = next(iter(valloader))
#     output = float_model(images)

# # convert to quantized model
torch.quantization.convert(quant_model, inplace=True)

print(quant_model.res1.net[0])
evaluate(quant_model, valloader, criterion)
# print("-"*80)
# print("test accuracy pre-quantisation")
# evaluate(float_model, testloader, criterion)
# print("-"*80)
# print("test accuracy post-quantisation")

# evaluate(quant_model, testloader, criterion)

# start = timer()
with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            float_outputs = float_model(inputs)
            quant_outputs = quant_model(inputs)
            break
# print(f"float time taken: {timer() - start}")
print("float\n", float_outputs[0])
print("quant\n", quant_outputs[0])

# start = timer()
# with torch.no_grad():
#         for data in testloader:
#             inputs, labels = data
#             outputs = quant_model(inputs)
#             break
# print(f"quant time taken: {timer() - start}")
# print(outputs[0])