import torch
from image_resnet_transfer_classifier import ImageResNetTransferClassifier
from torchvision.transforms import transforms
from PIL import Image
from torch.autograd import Variable
import data
import torch.nn as nn
import model
import torchvision
from torchvision.datasets import ImageFolder
import image_augmentations as ia
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Default values
default_test_folder = 'data/test'
default_weights_file = 'model.pt'

# Command-line arguments
test_folder = sys.argv[1] if len(sys.argv) > 1 else default_test_folder
weights_file = sys.argv[2] if len(sys.argv) > 2 else default_weights_file

batch_size = 35
divfac = 4
resize_size = (2048//divfac, 2048//divfac)

xfm_test = transforms.Compose([
                          transforms.CenterCrop((2048,2048)),
                          transforms.Resize(resize_size),
                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


xfm_test2 = transforms.Compose([
                            transforms.Resize(resize_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


test_dataset = ImageFolder(root=test_folder, transform=xfm_test2)

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, num_workers=0)

classifier = ImageResNetTransferClassifier(num_classes=len(test_dataset.classes))

print(test_dataset.classes)

if Path(weights_file).exists():
  if weights_file.endswith('.pt'):
    device = torch.device("cpu")
    classifier = torch.jit.load(weights_file)
  else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier.load_weights(Path(weights_file))


print(device)

# get some random training images
dataiter = iter(testloader)
images, labels = next(dataiter)

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
  all_labels = []
  all_preds = []
  for _, data in enumerate(testloader):
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
          index = topk.indices.cpu().numpy()[0]

          if index >= len(test_dataset.classes):
            index = len(test_dataset.classes) - 1
            print("Index out of bounds: ", index)

          actual = test_dataset.classes[index]
          all_labels.append(expected)
          all_preds.append(actual)
          if expected == actual:
            success += 1
          else:
            print("%s -> %s" % (expected, actual))
            failure += 1

  accuracy = success / (success + failure)
  print('Test [%d] loss: %.3f Success: %d Failure: %d Accuracy: %.3f Total: %d' %
              (epoch + 1, test_loss / total_items, success, failure, accuracy, success + failure))
  
  # Compute confusion matrix
  cm = confusion_matrix(all_labels, all_preds, labels=test_dataset.classes)
  print("Confusion Matrix:\n", cm)
  
  # Plot confusion matrix
  plt.figure(figsize=(10, 8))
  sns.heatmap(cm, annot=True, fmt='d', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes, cmap='Blues')
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.title('Confusion Matrix')
  plt.show()
  
  # Compute and print classification report
  report = classification_report(all_labels, all_preds, target_names=test_dataset.classes)
  print("Classification Report:\n", report)
  
  return accuracy

test_model(0)