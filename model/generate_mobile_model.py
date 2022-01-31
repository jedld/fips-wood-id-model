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
import csv
import json
import zipfile
import datetime


classifier, classnames = model.get('class_labels.txt','wts2.pth')
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
torchscript_model = optimize_for_mobile(torchscript_model)
print("generating mobile model pytorch version:", torch.__version__)
torch.jit.save(torchscript_model, "fips_wood_model_mobile.pt")

import csv

database_content = {}
with open('species_database.csv') as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  line_count = 0
  for row in csv_reader:
      if line_count == 0:
          print(f'Column names are {", ".join(row)}')
          line_count += 1
      else:
          line_count += 1
          key = row[0].lower().replace(' ', '_')
          database_content[key] = {
            'scientific_name': row[0].replace('_',' ').capitalize(),
            'other_names': [row[1]]
          }
  print(f'Processed {line_count} lines.')

with open('species_database.json', 'w') as outfile:
  json.dump(database_content, outfile, indent=2)

print("Building mobile asset archive...")

x = datetime.datetime.now()
model_version = x.strftime('%Y%m%d%H%M%S')
model_info = {
  'name' : 'wood-id-model',
  'description' : 'wood id model',
  'version' : model_version,
  'pytorch' : torch.__version__,
  'input_dimension' : input.shape
}

with open('model.json', 'w') as outfile:
  json.dump(model_info, outfile, indent=2)

zipf = zipfile.ZipFile(f'model-{model_version}.zip', 'w', zipfile.ZIP_DEFLATED)
zipf.write('species_database.json', 'species_database.json')
zipf.write('fips_wood_model_mobile.pt', 'model.pt')
zipf.write('model.txt','model.txt')
zipf.write('model.json','model.json')
zipf.write('class_labels.txt','labels.txt')
zipf.close()

print(f"model written to model-{model_version}.zip")