import torch
from image_resnet_transfer_classifier import ImageResNetTransferClassifier
from torchvision.transforms import transforms
from PIL import Image
from torch.autograd import Variable
import data
import torch.nn as nn
import model
import image_augmentations as ia

classifier, classnames = model.get('class_labels.txt','wts2.pth')
classifier.eval()

image = Image.open('..\\imgdb\\Train\\Hevea brasiliensis\\capture_20211021081401.jpg')
divfac = 4
resize_size = (2048//divfac, 2048//divfac)
# xfm = transforms.Compose([ia.PadToEnsureSize(out_size=(2048, 2048)),
#                           ia.Resize(out_size=resize_size),
#                           ia.ToTensor(),
#                           ia.ImageNetNormalize()])

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