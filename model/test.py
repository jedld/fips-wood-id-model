import torch
from image_resnet_transfer_classifier import ImageResNetTransferClassifier
from torchvision.transforms import transforms
from PIL import Image
from torch.autograd import Variable
import data
import torch.nn as nn
import model
import image_augmentations as ia
import sys

# Default values
default_labels_file = 'class_labels.txt'
default_weights_file = 'wts2.pth'

# Command-line arguments
labels_file = sys.argv[2] if len(sys.argv) > 2 else default_labels_file
weights_file = sys.argv[3] if len(sys.argv) > 3 else default_weights_file

classifier, classnames = model.get(labels_file, weights_file)
classifier.eval()

image = Image.open(sys.argv[1])
divfac = 4
resize_size = (2048//divfac, 2048//divfac)

xfm = transforms.Compose([ia.PadToEnsureSize(out_size=(2048, 2048)),
                           ia.Resize(out_size=resize_size),
                           ia.ToTensor(),
                           ia.ImageNetNormalize()])
sample = {'image': (image, ia.SampElemType.IMAGE)}
sample = xfm(sample)

input = sample['image'][0].unsqueeze(0)
print("input dimension", input.shape)

output = classifier(input)
output = nn.Softmax(dim=1)(output)

topk = torch.topk(output, len(classnames))
print("output ", classnames[topk.indices[0][0]])