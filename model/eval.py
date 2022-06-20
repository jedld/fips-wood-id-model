import torch
from image_resnet_transfer_classifier import ImageResNetTransferClassifier
from torchvision.transforms import transforms
from PIL import Image
from torch.autograd import Variable
import data
import torch.nn as nn
import model
import torchvision
from torchvision import *
import image_augmentations as ia
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

batch_size = 35
divfac = 4
resize_size = (2048//divfac, 2048//divfac)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

xfm_test = transforms.Compose([
                          transforms.CenterCrop((2048,2048)),
                          transforms.Resize(resize_size),
                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


xfm_test2 = transforms.Compose([
                            transforms.Resize(resize_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])                            



test_dataset = datasets.ImageFolder(root='../imgdb2/Test', transform=xfm_test2)



testloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, num_workers=0)

classifier = ImageResNetTransferClassifier(num_classes=len(test_dataset.classes))

print(test_dataset.classes)

PATH = './wts2.pth'

if Path(PATH).exists():
  classifier.load_weights(Path(PATH))

# get some random training images
dataiter = iter(testloader)
images, labels = dataiter.next()

# show images
print(' '.join('%5s' % test_dataset.classes[labels[j]] for j in range(10)))

classifier.eval()

classifier = classifier.to(device)

criterion = nn.CrossEntropyLoss()

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Get a batch of training data
inputs, classes = next(iter(testloader))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

# imshow(out, title=[test_dataset.classes[x] for x in classes])

def test_model(epoch):
  test_loss = 0.0
  total_items = 0
  success = 0
  failure = 0
  classifier.eval()
  for i, data in enumerate(testloader):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data[0].to(device), data[1].to(device)
      label_indexes = data[1].numpy()
      with torch.set_grad_enabled(False):
        outputs = classifier(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        total_items += 1

        for i2, out in enumerate(outputs):
          topk = torch.topk(out, len(test_dataset.classes))
          expected = test_dataset.classes[label_indexes[i2]]
          actual = test_dataset.classes[topk.indices.cpu().numpy()[0]]
          if  expected == actual:
            success+=1
          else:
            print("%s -> %s" % (expected, actual))
            failure+=1

  print('Test [%d] loss: %.3f Success: %d Failure: %d Accuracy: %.3f Total: %d' %
              (epoch + 1, test_loss / total_items, success, failure, success / (success + failure), success + failure))
  return success / (success + failure)

test_model(0)

