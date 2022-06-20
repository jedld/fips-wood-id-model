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
import os




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



print("Building mobile asset archive...")

x = datetime.datetime.now()
model_version = x.strftime('%Y%m%d%H%M%S')
model_info = {
  'name' : 'resnet-18-31-class',
  'description' : '31 class resnet50 Wood Model V5',
  'version' : model_version,
  'pytorch' : torch.__version__,
  'input_dimension' : input.shape,
  'accuracy_cutoff' : 4.0
}

with open('model.json', 'w') as outfile:
  json.dump(model_info, outfile, indent=2)

zipf = zipfile.ZipFile(f'model-{model_version}.zip', 'w', zipfile.ZIP_DEFLATED)
zipf.write('species_database.json', 'species_database.json')
zipf.write('fips_wood_model_mobile.pt', 'model.pt')
zipf.write('model.txt','model.txt')
zipf.write('model.json','model.json')
zipf.write('class_labels.txt','labels.txt')

reference_images = []
print(database_content.keys())
for subdir, dirs, files in os.walk('reference'):
  for file in files:
    #print os.path.join(subdir, file)
    filepath = subdir + os.sep + file
    reference_images.append([filepath, filepath.replace(' ', '_')])
    class_name = subdir.lower().replace(' ', '_').replace('reference/', '')
    if class_name not in database_content:
      print(f"{class_name} is not in the species_database.csv file!")
    else:
      content_detail = database_content[class_name]

      if 'reference_images' not in content_detail:
        content_detail['reference_images'] = []

      content_detail['reference_images'].append(filepath.replace(' ', '_'))

with open('species_database.json', 'w') as outfile:
  json.dump(database_content, outfile, indent=2)

for filepath, transformed_filepath in reference_images:
  zipf.write(filepath, transformed_filepath)

zipf.close()

print(f"model written to model-{model_version}.zip")
