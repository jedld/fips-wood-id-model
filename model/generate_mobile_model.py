import torch
from image_resnet_transfer_classifier import ImageResNetTransferClassifier
from torchvision.transforms import transforms
from torch.utils.mobile_optimizer import optimize_for_mobile
from PIL import Image
from torch.autograd import Variable
import data
import torch.nn as nn
import model
import image_augmentations as ia

classifier, classnames = model.get('class_labels.txt','wts.pth')
classifier.eval()

image = Image.open('../imgdb/Test/Acacia auriculiformis/Auri.5x.FPRDI.Authentic (1-1).jpg')

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

torchscript_model = torch.jit.script(classifier)
torchscript_model_optimized = optimize_for_mobile(torchscript_model)
print("generating mobile model pytorch version:", torch.__version__)
torch.jit.save(torchscript_model_optimized, "fips_wood_model_mobile.pt")
