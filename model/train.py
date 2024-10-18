import torch
from image_resnet_transfer_classifier import ImageResNetTransferClassifier
from torchvision.transforms import transforms
from PIL import Image
from torch.autograd import Variable
import data
import torch.nn as nn
import model
import torchvision
from torchvision import datasets
import image_augmentations as ia
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
  batch_size = 10
  divfac = 4
  resize_size = (2048//divfac, 2048//divfac)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  print(device)

  print("threads %d" % (torch.get_num_threads()))

  xfm3 = transforms.Compose([
                            transforms.Resize(resize_size),
                            transforms.RandomRotation(270),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

  xfm_test2 = transforms.Compose([
                            transforms.Resize(resize_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


  train_dataset = datasets.ImageFolder(root='data/train', transform=xfm3)

  test_dataset = datasets.ImageFolder(root='data/test', transform=xfm_test2)

  fused_trainset = torch.utils.data.ConcatDataset([train_dataset])
  fused_testset = torch.utils.data.ConcatDataset([test_dataset])


  trainloader = torch.utils.data.DataLoader(fused_trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
  testloader = torch.utils.data.DataLoader(fused_testset, batch_size=10, num_workers=2)



  total_classes = len(train_dataset.classes)
  print('Total classes %s' % (total_classes ))
  classifier = ImageResNetTransferClassifier(num_classes=total_classes)


  if (train_dataset.classes != test_dataset.classes):
    print('Error test and train do not have the same classes')

  print(train_dataset.classes)

  PATH = './wts2.pth'

  if Path(PATH).exists():
    classifier.load_weights(Path(PATH))

  # get some random training images
  dataiter = iter(trainloader)
  images, labels = next(dataiter)

  # show images
  print(' '.join('%5s' % train_dataset.classes[labels[j]] for j in range(batch_size)))

  with open('class_labels.txt', 'w') as the_file:
      for c in train_dataset.classes:
            the_file.write(c + '\n')

  classifier.eval()

  classifier = classifier.to(device)

  criterion = nn.CrossEntropyLoss()

  optimizer = optim.SGD(classifier.parameters(), lr=0.0001, momentum=0.9)


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
  inputs, classes = next(iter(trainloader))

  # Make a grid from batch
  out = torchvision.utils.make_grid(inputs)

  # imshow(out, title=[train_dataset.classes[x] for x in classes])

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
              failure+=1

    print('Test [%d] loss: %.3f Success: %d Failure: %d Accuracy: %.3f Total: %d' %
                (epoch + 1, test_loss / total_items, success, failure, success / (success + failure), success + failure))
    return success / (success + failure)

  best_accuracy = test_model(0)

  # Start model training
  for epoch in range(100):  # loop over the dataset multiple times
      running_loss = 0.0
      classifier.train()
      success = 0
      failure = 0
      for i, data in enumerate(trainloader, 0):
          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data[0].to(device), data[1].to(device)

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = classifier(inputs)
        
          label_indexes = data[1].numpy()
          for i2, out in enumerate(outputs):
            topk = torch.topk(out, len(train_dataset.classes))
            expected = train_dataset.classes[label_indexes[i2]]
            actual = train_dataset.classes[topk.indices.cpu().numpy()[0]]
            if expected == actual:
              success+=1
            else:
              failure+=1


          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          # print statistics
          running_loss += loss.item()
          # print every 10 mini-batches

          print('[%d, %5d] loss: %.3f acc: %.3f best: %.3f ' %
                (epoch + 1, i + 1, running_loss / 10, success / (success + failure), best_accuracy))
          running_loss = 0.0

      accuracy = test_model(epoch)
      if (accuracy > best_accuracy):
        print("best accuracy %f" % (accuracy))
        best_accuracy = accuracy
        classifier.save_weights(PATH)
        with open('model.txt', 'w') as the_file:
          the_file.write('%.3f' % (accuracy))


  print('Finished Training. Best accuracy %.3f' % (best_accuracy))


main()

# report on the F1 and accuracy scores 
