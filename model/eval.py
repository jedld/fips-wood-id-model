import torch
from image_resnet_transfer_classifier import ImageResNetTransferClassifier
from torchvision.transforms import transforms
from PIL import Image
from torch.autograd import Variable
import data
import torch.nn as nn
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
from optparse import OptionParser
from stn_model import MinimalCNN
from Resnet18 import ResNet18
from mobilenet import MobileNetV2TransferClassifier
import json
import ast
from collections import defaultdict
from itertools import combinations

from torchsummary import summary
NORMALIZATION_STATS = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

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
    except FileNotFoundError:
        print(f"Warning: {model_txt_path} not found. Using default settings.")
    return settings

# Command-line options
parser = OptionParser()
parser.add_option("-t", "--test_folder", dest="test_folder", default="data/test",
                  help="path to the test folder", metavar="FOLDER")
parser.add_option("-w", "--weights_file", dest="weights_file", default="checkpoint.pth",
                  help="path to the model weights file", metavar="FILE")
parser.add_option("-g", "--grayscale", dest="grayscale", default=False,
                  help="use grayscale images")
parser.add_option("-m", "--model_txt", dest="model_txt", default="model.txt",
                  help="path to model.txt file", metavar="FILE")

(options, args) = parser.parse_args()

# Default values
test_folder = options.test_folder
weights_file = options.weights_file

# Read settings from model.txt
settings = parse_model_txt(options.model_txt)

# Use settings from model.txt if available, otherwise use defaults
input_size = settings.get('input_size', 2048)
is_grayscale = settings.get('grayscale', options.grayscale)
normalization_stats = settings.get('normalization_stats', NORMALIZATION_STATS)
model_type = settings.get('model_type', 'mobilenet')

batch_size = 35
divfac = 4
resize_size = (input_size//divfac, input_size//divfac)

# use normalization from model.txt or default to imagenet normalization
normalize = transforms.Normalize(mean=normalization_stats[0], std=normalization_stats[1])

if is_grayscale:
    xfm_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.CenterCrop((input_size, input_size)),
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        normalize
    ])

    xfm_test2 = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        normalize
    ])
else:
    xfm_test = transforms.Compose([
        transforms.CenterCrop((input_size, input_size)),
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        normalize
    ])

    xfm_test2 = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        normalize
    ])

test_dataset = ImageFolder(root=test_folder, transform=xfm_test2)

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, num_workers=0)

# Initialize model based on type from model.txt
if model_type == 'minimalcnn':
    classifier = MinimalCNN(num_classes=len(test_dataset.classes))
elif model_type == 'mobilenet':
    classifier = MobileNetV2TransferClassifier(num_classes=len(test_dataset.classes))
elif model_type == 'resnet18':
    classifier = ImageResNetTransferClassifier(num_classes=len(test_dataset.classes))
else:
    print(f"Warning: Unknown model type {model_type}, defaulting to MobileNetV2")
    classifier = MobileNetV2TransferClassifier(num_classes=len(test_dataset.classes))

print(f"Using model type: {model_type}")
print(f"Input size: {input_size}x{input_size}")
print(f"Grayscale: {is_grayscale}")
print(f"Normalization stats: {normalization_stats}")

print(test_dataset.classes)

if Path(weights_file).exists():
  if weights_file.endswith('.pt'):
    device = torch.device("cpu")
    classifier = torch.jit.load(weights_file)
  else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier.load_weights(Path(weights_file))
else:
  print("Weights file not found: ", weights_file)
  sys.exit(1)


print(device)
# summary(classifier, input_size=(3, 512, 512))
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

def analyze_confusion_matrix(cm, class_names, threshold=0.3):
    """
    Analyze confusion matrix to suggest class merges.
    Args:
        cm: Confusion matrix
        class_names: List of class names
        threshold: Minimum confusion ratio to consider classes for merging
    Returns:
        List of suggested class merges with their confusion ratios
    """
    n_classes = len(class_names)
    confusion_ratios = []
    
    # Calculate confusion ratios for each pair of classes
    for i, j in combinations(range(n_classes), 2):
        # Calculate confusion ratio in both directions
        total_i = cm[i].sum()
        total_j = cm[j].sum()
        
        if total_i == 0 or total_j == 0:
            continue
            
        # Ratio of class i being confused as class j
        ratio_i_to_j = cm[i, j] / total_i
        # Ratio of class j being confused as class i
        ratio_j_to_i = cm[j, i] / total_j
        
        # Average confusion ratio
        avg_ratio = (ratio_i_to_j + ratio_j_to_i) / 2
        
        if avg_ratio >= threshold:
            confusion_ratios.append({
                'class1': class_names[i],
                'class2': class_names[j],
                'ratio': avg_ratio,
                'i_to_j': ratio_i_to_j,
                'j_to_i': ratio_j_to_i,
                'total_samples': total_i + total_j
            })
    
    # Sort by confusion ratio
    confusion_ratios.sort(key=lambda x: x['ratio'], reverse=True)
    return confusion_ratios

def suggest_class_merges(confusion_ratios, min_ratio=0.3, min_samples=10):
    """
    Suggest class merges based on confusion analysis.
    Args:
        confusion_ratios: List of confusion ratios from analyze_confusion_matrix
        min_ratio: Minimum confusion ratio to suggest a merge
        min_samples: Minimum number of samples required to consider a merge
    Returns:
        List of suggested merges with explanations
    """
    suggestions = []
    used_classes = set()
    
    for conf in confusion_ratios:
        if conf['ratio'] < min_ratio or conf['total_samples'] < min_samples:
            continue
            
        class1, class2 = conf['class1'], conf['class2']
        
        # Skip if either class is already used in a merge
        if class1 in used_classes or class2 in used_classes:
            continue
            
        suggestion = {
            'classes': [class1, class2],
            'confusion_ratio': conf['ratio'],
            'explanation': (
                f"Classes '{class1}' and '{class2}' show high confusion "
                f"({conf['ratio']:.2%}). Class '{class1}' is confused as '{class2}' "
                f"{conf['i_to_j']:.2%} of the time, and '{class2}' is confused as '{class1}' "
                f"{conf['j_to_i']:.2%} of the time."
            )
        }
        suggestions.append(suggestion)
        used_classes.add(class1)
        used_classes.add(class2)
    
    return suggestions

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
    
    # Analyze confusion matrix and suggest merges
    confusion_ratios = analyze_confusion_matrix(cm, test_dataset.classes)
    merge_suggestions = suggest_class_merges(confusion_ratios)
    
    if merge_suggestions:
        print("\nSuggested Class Merges:")
        print("=======================")
        for suggestion in merge_suggestions:
            print(f"\nSuggestion: Merge {suggestion['classes']}")
            print(f"Confusion Ratio: {suggestion['confusion_ratio']:.2%}")
            print(f"Explanation: {suggestion['explanation']}")
        
        # Save suggestions to file
        suggestions_file = "class_merge_suggestions.txt"
        with open(suggestions_file, "w") as f:
            f.write("Class Merge Suggestions\n")
            f.write("======================\n\n")
            for suggestion in merge_suggestions:
                f.write(f"Suggestion: Merge {suggestion['classes']}\n")
                f.write(f"Confusion Ratio: {suggestion['confusion_ratio']:.2%}\n")
                f.write(f"Explanation: {suggestion['explanation']}\n\n")
        print(f"\nDetailed suggestions saved to {suggestions_file}")
    
    # Compute and print classification report
    report = classification_report(all_labels, all_preds, target_names=test_dataset.classes)
    print("Classification Report:\n", report)
    # write report to file
    if options.grayscale:
        with open("classification_report_grayscale.txt", "w") as file:
            file.write(report)
    else:
        with open("classification_report.txt", "w") as file:
            file.write(report)
    
    return accuracy

test_model(0)