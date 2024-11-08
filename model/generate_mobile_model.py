import torch
from image_resnet_transfer_classifier import ImageResNetTransferClassifier
from torchvision.transforms import transforms
from torch.utils.mobile_optimizer import optimize_for_mobile
from PIL import Image
import model
import image_augmentations as ia
import csv
import json
import zipfile
import datetime
import os
from pathlib import Path
from optparse import OptionParser

# Command-line options
parser = OptionParser()
parser.add_option("-m", "--model", dest="model_file", default="wts.pth",
                  help="path to the model weights file", metavar="FILE")
parser.add_option("-c", "--class_labels", dest="class_labels_file", default="class_labels.txt",
                  help="path to the class labels file", metavar="FILE")
parser.add_option("-n", "--name", dest="model_name", default="resnet-18-31-class",
                  help="name of the model", metavar="NAME")
parser.add_option("-d", "--description", dest="model_description", default="31 class resnet50 Wood Model V5",
                  help="description of the model", metavar="DESCRIPTION")
parser.add_option("-t", "--type", dest="model_type", default="resnet18")
parser.add_option("-o", "--output", dest="output", default="model.zip")
parser.add_option("-p","--path", dest="path", default=".")

(options, args) = parser.parse_args()

# define and load the model here
def load_model(class_labels_file_path, model_file_path, arch='resnet18'):
    with open(class_labels_file_path, 'r') as fh:
      lines = [line.strip() for line in fh.readlines()]
    cls = ImageResNetTransferClassifier(body_arch=arch, num_classes=len(lines))
    cls.load_weights(Path(model_file_path))

    return cls, lines

x = datetime.datetime.now()
model_version = x.strftime('%Y%m%d%H%M%S')

# Load model and class
class_labels_file_path = os.path.join(options.path, options.class_labels_file)
model_file_path = os.path.join(options.path, options.model_file)

classifier, classnames = load_model(class_labels_file_path, model_file_path, arch=options.model_type)
classifier.eval()


divfac = 4
resize_size = (2048//divfac, 2048//divfac)
xfm = transforms.Compose([ia.PadToEnsureSize(out_size=(2048, 2048)),
                          ia.Resize(out_size=resize_size),
                          ia.ToTensor(),
                          ia.ImageNetNormalize()])


input = torch.rand(1, 3, resize_size[0], resize_size[1])
print("input dimension", input.shape)

# Convert model to TorchScript and optimize for mobile
torchscript_model = torch.jit.script(classifier)
torchscript_model = optimize_for_mobile(torchscript_model)
print("generating mobile model pytorch version:", torch.__version__)
torch.jit.save(torchscript_model, "tmp_fips_wood_model_mobile.pt")

# Load species database
database_content = {}

specied_database_file = os.path.join(options.path, 'species_database.csv')
species_info_json_file = os.path.join(options.path, 'species_database.json')

if not os.path.exists(specied_database_file):
  print("species_database.csv not found will generate a sample template. Until this is properly filled up labels in the mobile app will be equal to the class label which may not be user friendly.\n It is recommended to edit the csv and rerun this script.")

  with open(specied_database_file, 'w') as csv_file:
    csv_file.write("scientific_name,other_names\n")
    for class_name in classnames:
      csv_file.write(f"{class_name},{class_name}\n")

  print(f"species_database.csv generated and placed at {specied_database_file}. Please edit the file and rerun this script.")

if not os.path.exists(species_info_json_file):
  print(f"species_database.json not found will generate a sample template. Until this is properly filled up labels in the mobile app will be equal to the class label which may not be user friendly.\n It is recommended to edit the json and rerun this script.")

  with open(species_info_json_file, 'w') as json_file:
    json.dump({}, json_file)

  print(f"species_database.json generated and placed at {species_info_json_file}. Please edit the file and rerun this script.")

model_txt_file = os.path.join(options.path, 'model.txt')

if not os.path.exists(model_txt_file):
  print("Generating model.txt place version string here and rerun this script.")
  with open(model_txt_file, 'w') as txt_file:
    current_date = datetime.datetime.now()
    txt_file.write(f"{current_date.strftime('%Y%m%d%H%M%S')}")

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

model_info = {
  'name' : options.model_name,
  'description' : options.model_description,
  'version' : model_version,
  'pytorch' : torch.__version__,
  'input_dimension' : input.shape,
  'accuracy_cutoff' : 0.6
}

with open('model.json', 'w') as outfile:
  json.dump(model_info, outfile, indent=2)

reference_images = []
print(database_content.keys())
if not os.path.exists('reference'):
  print("reference directory not found. Please place the reference images in the reference directory and rerun this script.")
  os.makedirs('reference')
  for class_name in database_content.keys():
    os.makedirs(f'reference/{class_name}', exist_ok=True)

for subdir, _, files in os.walk('reference'):
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
output_file = os.path.join(options.path, options.output)
zipf = zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED)
zipf.write('species_database.json', 'species_database.json')
zipf.write("tmp_fips_wood_model_mobile.pt", 'model.pt')
zipf.write('model.txt','model.txt')
zipf.write('model.json','model.json')
zipf.write('class_labels.txt','labels.txt')

for filepath, transformed_filepath in reference_images:
  zipf.write(filepath, transformed_filepath)

zipf.close()

# delete temp files
os.remove("tmp_fips_wood_model_mobile.pt")

print(f"model written to {output_file}")