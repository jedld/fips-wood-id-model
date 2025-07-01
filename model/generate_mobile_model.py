import torch
from image_resnet_transfer_classifier import ImageResNetTransferClassifier
from mobilenet import MobileNetV2TransferClassifier
from efficientnet import EfficientNetTransferClassifier
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
import ast

# Command-line options
parser = OptionParser()
parser.add_option("-m", "--model", dest="model_file", default="checkpoint.pth",
                  help="path to the model weights file", metavar="FILE")
parser.add_option("-c", "--class_labels", dest="class_labels_file", default="class_labels.txt",
                  help="path to the class labels file", metavar="FILE")
parser.add_option("-n", "--name", dest="model_name", default=None,
                  help="name of the model", metavar="NAME")
parser.add_option("-d", "--description", dest="model_description", default="Wood Identification Model",
                  help="description of the model", metavar="DESCRIPTION")
parser.add_option("-t", "--type", dest="model_type", default='efficientnet',
                  help="model type (mobilenet, resnet18, efficientnet)", metavar="TYPE")
parser.add_option("-o", "--output", dest="output", default="model.zip")
parser.add_option("-p", "--path", dest="path", default=".")
parser.add_option("-s", "--size", dest="input_size", default=512, type="int", 
                  help="input size (512 or 1024)", metavar="SIZE")
parser.add_option("--swa", dest="use_swa", default=False, action="store_true",
                  help="use SWA model weights", metavar="BOOL")
parser.add_option("--checkpoint", dest="checkpoint",
                  help="checkpoint name (without extension) to use", metavar="NAME")

(options, args) = parser.parse_args()

def parse_model_txt(model_txt_path):
    """Parse model.txt file and return settings as a dictionary."""
    settings = {}
    try:
        with open(model_txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Handle special cases
                    if key == 'Input Size':
                        size = int(value.split('x')[0])
                        settings['input_size'] = size
                    elif key == 'Model Type':
                        settings['model_type'] = value
                    elif key == 'Grayscale':
                        settings['grayscale'] = value.lower() == 'true'
                    elif key == 'Normalization Stats':
                        # Convert string representation of tuple to actual tuple
                        stats = ast.literal_eval(value)
                        settings['normalization_stats'] = stats
                    elif key == 'Model Name':
                        settings['model_name'] = value
                    elif key == 'Description':
                        settings['description'] = value
    except FileNotFoundError:
        print(f"Warning: {model_txt_path} not found. Using default settings.")
    return settings

# Handle checkpoint and model info paths
if options.checkpoint:
    checkpoint_dir = Path("checkpoints")
    model_file = checkpoint_dir / f"{options.checkpoint}.pth"
    if options.use_swa:
        model_file = checkpoint_dir / f"{options.checkpoint}.pth.swa.final"
    model_txt = checkpoint_dir / f"{options.checkpoint}.txt"
    
    # Read settings from model.txt
    settings = parse_model_txt(model_txt)
    
    # Use settings from model.txt if available
    if not options.model_type and 'model_type' in settings:
        options.model_type = settings['model_type']
    if not options.model_name and 'model_name' in settings:
        options.model_name = settings['model_name']
    if not options.model_description and 'description' in settings:
        options.model_description = settings['description']
    if 'input_size' in settings:
        options.input_size = settings['input_size']
else:
    model_file = options.model_file
    model_txt = None

# Generate model name with timestamp if not provided
if not options.model_name:
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    options.model_name = f"{options.model_type or 'model'}_{timestamp}"

# define and load the model here
def load_model(class_labels_file_path, model_file_path, arch='mobilenet'):
    """
    Load the model and class labels.
    
    Args:
        class_labels_file_path (str): Path to the class labels file
        model_file_path (str): Path to the model file
        arch (str): Model architecture to use
        
    Returns:
        tuple: (model, class_names)
    """
    # Load class labels
    with open(class_labels_file_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"class_names: {class_names}")
    # Initialize model based on architecture
    if arch == 'mobilenet':
        model = MobileNetV2TransferClassifier(num_classes=len(class_names))
    elif arch == 'resnet18':
        model = ImageResNetTransferClassifier(num_classes=len(class_names))
    elif arch == 'efficientnet':
        model = EfficientNetTransferClassifier(num_classes=len(class_names))
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    # Load model weights
    if str(model_file_path).endswith('.pt'):
        # Load TorchScript model
        model = torch.jit.load(model_file_path)
    else:
        # Load PyTorch model
        model.load_weights(model_file_path)
    
    model.eval()
    return model, class_names

# Load model and class
class_labels_file_path = os.path.join(options.path, options.class_labels_file)
print(f"class_labels_file_path: {class_labels_file_path}")
print(f"model_file: {model_file}")
classifier, classnames = load_model(class_labels_file_path, model_file, arch=options.model_type)
classifier.eval()

print(f"Model Information:")
print(f"=================")
print(f"Model Type: {options.model_type}")
print(f"Model Name: {options.model_name}")
print(f"Input Size: {options.input_size}x{options.input_size}")
print(f"Using {'SWA' if options.use_swa else 'regular'} model weights")
print(f"=================")

resize_size = (options.input_size, options.input_size)
xfm = transforms.Compose([
    ia.PadToEnsureSize(out_size=(options.input_size, options.input_size)),
    ia.Resize(out_size=resize_size),
    ia.ToTensor(),
    ia.ImageNetNormalize()
])

input = torch.rand(1, 3, resize_size[0], resize_size[1])
print("input dimension", input.shape)

# Convert model to TorchScript and optimize for mobile
print("Converting model to TorchScript...")
torchscript_model = torch.jit.script(classifier)
print("Optimizing for mobile...")
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

# Generate model.txt if not exists
model_txt_file = os.path.join(options.path, 'model.txt')
if not os.path.exists(model_txt_file):
    print("Generating model.txt...")
    with open(model_txt_file, 'w') as txt_file:
        current_date = datetime.datetime.now()
        txt_file.write(f"Version: {current_date.strftime('%Y%m%d%H%M%S')}\n")
        txt_file.write(f"Model Name: {options.model_name}\n")
        txt_file.write(f"Model Type: {options.model_type}\n")
        txt_file.write(f"Input Size: {options.input_size}x{options.input_size}\n")
        txt_file.write(f"Model Description: {options.model_description}\n")
        txt_file.write(f"PyTorch Version: {torch.__version__}\n")
        txt_file.write(f"Generated Date: {current_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if options.use_swa:
            txt_file.write(f"Using SWA Model: True\n")

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
    'name': options.model_name,
    'description': options.model_description,
    'version': datetime.datetime.now().strftime('%Y%m%d%H%M%S'),
    'pytorch': torch.__version__,
    'input_dimension': input.shape,
    'accuracy_cutoff': 0.6,
    'model_type': options.model_type,
    'using_swa': options.use_swa
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
print(f"Creating model package: {output_file}")
zipf = zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED)
zipf.write('species_database.json', 'species_database.json')
zipf.write("tmp_fips_wood_model_mobile.pt", 'model.pt')
zipf.write('model.txt', 'model.txt')
zipf.write('model.json', 'model.json')
zipf.write('class_labels.txt', 'labels.txt')

for filepath, transformed_filepath in reference_images:
    zipf.write(filepath, transformed_filepath)

zipf.close()

# delete temp files
os.remove("tmp_fips_wood_model_mobile.pt")

print(f"Model package written to {output_file}")