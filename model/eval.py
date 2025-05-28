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
from efficientnet import EfficientNetTransferClassifier
import json
import ast
from collections import defaultdict
from itertools import combinations
import os
import pandas as pd
import cv2
from torch.nn import functional as F

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
parser.add_option("-w", "--weights_file", dest="weights_file", default=None,
                  help="path to the model weights file", metavar="FILE")
parser.add_option("-g", "--grayscale", dest="grayscale", default=False,
                  help="use grayscale images")
parser.add_option("-m", "--model_txt", dest="model_txt", default=None,
                  help="path to model.txt file", metavar="FILE")
parser.add_option("-M", "--model", dest="model", default="efficientnet",
                  help="model to use", metavar="MODEL")
parser.add_option("-o", "--output", dest="output_file", default="class_performance_report.png",
                  help="path to save the performance report", metavar="FILE")
parser.add_option("-c", "--checkpoint", dest="checkpoint", default=None,
                  help="checkpoint name (without extension) to use", metavar="NAME")
parser.add_option("-s", "--swa", dest="use_swa", default=False,
                  help="use SWA model weights", metavar="BOOL")

(options, args) = parser.parse_args()

# Default values
test_folder = options.test_folder

# Handle checkpoint and model info paths
if options.checkpoint:
    checkpoint_dir = Path("checkpoints")
    weights_file = checkpoint_dir / f"{options.checkpoint}.pth"
    if options.use_swa:
        weights_file = checkpoint_dir / f"{options.checkpoint}.pth.swa.final"
    model_txt = checkpoint_dir / f"{options.checkpoint}.txt"
else:
    weights_file = options.weights_file
    model_txt = options.model_txt

if weights_file is None:
    print("Error: Either --weights_file or --checkpoint must be specified")
    sys.exit(1)

# Read settings from model.txt
settings = parse_model_txt(model_txt)

# Use settings from model.txt if available, otherwise use defaults
is_grayscale = settings.get('grayscale', options.grayscale)
normalization_stats = settings.get('normalization_stats', NORMALIZATION_STATS)
model_type = settings.get('model_type', options.model)

# Print model information
print("\nModel Information:")
print("=================")
if Path(model_txt).exists():
    with open(model_txt, 'r') as f:
        print(f.read())
print("=================\n")
print(f"Using {'SWA' if options.use_swa else 'regular'} model weights")

batch_size = 35
divfac = 4
resize_size = (2048//divfac, 2048//divfac)

# use normalization from model.txt or default to imagenet normalization
normalize = transforms.Normalize(mean=normalization_stats[0], std=normalization_stats[1])

if is_grayscale:
    xfm_test = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.CenterCrop((2048, 2048)),
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
        transforms.CenterCrop((2048, 2048)),
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        normalize
    ])

    xfm_test2 = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        normalize
    ])

test_dataset = ImageFolder(root=test_folder, transform=xfm_test)

testloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, num_workers=0)

# Initialize model based on type from model.txt
if model_type == 'minimalcnn':
    classifier = MinimalCNN(num_classes=len(test_dataset.classes))
elif model_type == 'mobilenet':
    classifier = MobileNetV2TransferClassifier(num_classes=len(test_dataset.classes))
elif model_type == 'resnet18':
    classifier = ImageResNetTransferClassifier(num_classes=len(test_dataset.classes))
elif model_type == 'efficientnet':
    classifier = EfficientNetTransferClassifier(num_classes=len(test_dataset.classes))
else:
    print(f"Warning: Unknown model type {model_type}, defaulting to MobileNetV2")
    classifier = MobileNetV2TransferClassifier(num_classes=len(test_dataset.classes))

print(f"Using model type: {model_type}")
print(f"Input size: 512x512")
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

def imshow(inp, ax=None, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    if ax is None:
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)
    else:
        ax.imshow(inp)
        if title is not None:
            ax.set_title(title)

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

def plot_confused_images(inputs, true_labels, pred_labels, class_names, test_dataset, threshold=0.3):
    """
    Plot pairs of images that were confused with each other and save them to a directory.
    Args:
        inputs: Batch of input images
        true_labels: True labels for the images
        pred_labels: Predicted labels for the images
        class_names: List of class names
        test_dataset: The test dataset to get sample images from
        threshold: Minimum confidence threshold to consider an image as confused
    """
    # Create output directory for confusion plots
    output_dir = Path("confusion_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Create a dictionary to store confused image pairs
    confused_pairs = defaultdict(list)
    
    # Get sample images for each class
    class_samples = defaultdict(list)
    for idx, (img, label) in enumerate(test_dataset):
        class_samples[label].append((img, test_dataset.samples[idx][0]))
        if len(class_samples[label]) >= 3:  # Store 3 samples per class
            break
    
    for i, (true_label, pred_label) in enumerate(zip(true_labels, pred_labels)):
        if true_label != pred_label:
            pair_key = tuple(sorted([true_label, pred_label]))
            confused_pairs[pair_key].append((inputs[i], true_label, pred_label))
    
    # Plot confused pairs
    for (class1, class2), pairs in confused_pairs.items():
        if len(pairs) > 0:
            print(f"\nConfused pairs between {class_names[class1]} and {class_names[class2]}:")
            n_pairs = min(len(pairs), 5)  # Show at most 5 pairs per confusion
            
            # Create figure with proper size and DPI
            fig, axes = plt.subplots(n_pairs, 4, figsize=(20, 3*n_pairs), dpi=100)
            fig.suptitle(f'Confusion between {class_names[class1]} and {class_names[class2]}', fontsize=16)
            
            for i, (img, true_label, pred_label) in enumerate(pairs[:n_pairs]):
                if n_pairs == 1:
                    ax1, ax2, ax3, ax4 = axes
                else:
                    ax1, ax2, ax3, ax4 = axes[i]
                
                # Plot confused image
                img_np = img.cpu().numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = std * img_np + mean
                img_np = np.clip(img_np, 0, 1)
                ax1.imshow(img_np)
                ax1.set_title(f'Confused Image\nTrue: {class_names[true_label]}', fontsize=10)
                ax1.axis('off')
                
                # Plot sample of true class
                if class_samples[true_label]:
                    sample_img, sample_path = class_samples[true_label][0]
                    sample_np = sample_img.numpy().transpose((1, 2, 0))
                    sample_np = std * sample_np + mean
                    sample_np = np.clip(sample_np, 0, 1)
                    ax2.imshow(sample_np)
                    ax2.set_title(f'Sample {class_names[true_label]}\n{Path(sample_path).name}', fontsize=10)
                    ax2.axis('off')
                
                # Plot sample of predicted class
                if class_samples[pred_label]:
                    sample_img, sample_path = class_samples[pred_label][0]
                    sample_np = sample_img.numpy().transpose((1, 2, 0))
                    sample_np = std * sample_np + mean
                    sample_np = np.clip(sample_np, 0, 1)
                    ax3.imshow(sample_np)
                    ax3.set_title(f'Sample {class_names[pred_label]}\n{Path(sample_path).name}', fontsize=10)
                    ax3.axis('off')
                
                # Add filename of confused image
                ax4.axis('off')
                ax4.text(0.1, 0.5, f'Filename:\n{test_dataset.samples[i][0]}', 
                        fontsize=8, wrap=True)
            
            plt.tight_layout()
            
            # Save the plot
            safe_class1 = class_names[class1].replace('/', '_')
            safe_class2 = class_names[class2].replace('/', '_')
            output_file = output_dir / f'confusion_{safe_class1}_vs_{safe_class2}.png'
            plt.savefig(output_file, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"Saved confusion plot to {output_file}")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def __call__(self, x, index=None):
        # Enable gradient computation
        x.requires_grad_(True)
        
        # Forward pass
        output = self.model(x)
        
        if index is None:
            index = output.argmax(dim=1)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Create one-hot encoding for the target class
        one_hot = torch.zeros_like(output)
        one_hot[0][index] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.detach().cpu()
        activations = self.activations.detach().cpu()
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3))
        
        # Create weighted activation map
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), 
                          size=x.shape[2:], 
                          mode='bilinear', 
                          align_corners=False)
        cam = cam.squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam

def get_target_layer(model, model_type):
    if model_type == 'mobilenet':
        return model.model.features[-1]
    elif model_type == 'resnet18':
        return model.model.layer4[-1]
    elif model_type == 'efficientnet':
        return model.model.conv_head
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def visualize_gradcam(model, input_tensor, original_image, class_idx, model_type, true_class, pred_class, confidence, save_path=None):
    # Get target layer for Grad-CAM
    target_layer = get_target_layer(model, model_type)
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Generate heatmap
    heatmap = grad_cam(input_tensor, class_idx)
    
    # Convert heatmap to RGB using a red color scheme
    heatmap = np.uint8(255 * heatmap)
    # Create a custom red colormap (dark red to light red)
    red_cmap = np.zeros((256, 1, 3), dtype=np.uint8)
    red_cmap[:, 0, 0] = np.linspace(0, 255, 256)  # Red channel
    red_cmap[:, 0, 1] = np.linspace(0, 200, 256)  # Green channel
    red_cmap[:, 0, 2] = np.linspace(0, 200, 256)  # Blue channel
    red_cmap = cv2.applyColorMap(red_cmap, cv2.COLORMAP_HOT)
    red_cmap = red_cmap.reshape(256, 3)
    
    # Apply the custom colormap
    heatmap_rgb = np.zeros((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        heatmap_rgb[:, :, i] = red_cmap[heatmap, i]
    
    # Convert original image to numpy array
    # denormalize the original image from resnet stats
    original_image = original_image.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    original_image = original_image * normalization_stats[1] + normalization_stats[0]
    original_image = (original_image * 255).astype(np.uint8)
    
    # Resize heatmap to match original image
    heatmap_rgb = cv2.resize(heatmap_rgb, (original_image.shape[1], original_image.shape[0]))
    
    # Blend heatmap with original image
    alpha = 0.6
    output = cv2.addWeighted(original_image, 1-alpha, heatmap_rgb, alpha, 0)
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot original image (without any overlay)
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title(f'Original Image\nTrue: {true_class}', fontsize=10)
    plt.axis('off')
    
    # Plot heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_rgb)
    plt.title('Grad-CAM Heatmap\nDark Red = Most Important', fontsize=10)
    plt.axis('off')
    
    # Plot blended image (heatmap overlay)
    plt.subplot(1, 3, 3)
    plt.imshow(output)
    plt.title(f'Heatmap Overlay\nPred: {pred_class}\nConf: {confidence:.2%}', fontsize=10)
    plt.axis('off')
    
    # Add overall title
    plt.suptitle(f'True: {true_class} → Pred: {pred_class} (Conf: {confidence:.2%})', 
                fontsize=12, y=1.05)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def test_model(epoch):
    test_loss = 0.0
    total_items = 0
    success = 0
    failure = 0
    classifier.eval()
    all_labels = []
    all_preds = []
    all_inputs = []
    all_true_labels = []
    all_pred_labels = []
    
    # Create directory for Grad-CAM visualizations
    gradcam_dir = Path("gradcam_visualizations")
    gradcam_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for each class
    for class_name in test_dataset.classes:
        class_dir = gradcam_dir / class_name
        class_dir.mkdir(exist_ok=True)
    
    for batch_idx, data in enumerate(testloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        label_indexes = data[1].numpy()
        
        # Enable gradient computation for Grad-CAM
        inputs.requires_grad_(True)
        
        with torch.set_grad_enabled(True):  # Enable gradients for Grad-CAM
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            total_items += 1

            for i2, out in enumerate(outputs):
                # Get top prediction and confidence
                probs = F.softmax(out, dim=0)
                topk = torch.topk(probs, len(test_dataset.classes))
                pred_idx = topk.indices[0].item()
                confidence = topk.values[0].item()
                
                expected = test_dataset.classes[label_indexes[i2]]
                actual = test_dataset.classes[pred_idx]
                
                all_labels.append(expected)
                all_preds.append(actual)
                all_inputs.append(inputs[i2].detach().cpu())
                all_true_labels.append(label_indexes[i2])
                all_pred_labels.append(pred_idx)
                
                # Generate Grad-CAM visualization for each image
                # Create descriptive filename with batch and image indices
                filename = f"img_{batch_idx}_{i2}_true_{expected}_pred_{actual}_conf_{confidence:.2f}.png"
                save_path = gradcam_dir / expected / filename
                
                visualize_gradcam(
                    classifier,
                    inputs[i2:i2+1],
                    inputs[i2],
                    pred_idx,
                    model_type,
                    expected,
                    actual,
                    confidence,
                    save_path
                )
                
                if expected == actual:
                    success += 1
                else:
                    print(f"{expected} -> {actual} (Confidence: {confidence:.2%})")
                    failure += 1

    # Plot confused images
    plot_confused_images(
        torch.stack(all_inputs),
        torch.tensor(all_true_labels),
        torch.tensor(all_pred_labels),
        test_dataset.classes,
        test_dataset
    )

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
    
    return accuracy, all_labels, all_preds

def generate_performance_report(y_true, y_pred, classes, output_file='class_performance_report.png'):
    """
    Generate a comprehensive performance report with visualizations.
    Each visualization is saved as a separate high-quality image.
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
        classes: List of class names
        output_file: Base path for output files (without extension)
    """
    # Calculate metrics
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Remove extension from output_file to use as base name
    base_output = os.path.splitext(output_file)[0]
    
    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix', fontsize=14, pad=20)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{base_output}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-class Metrics Bar Plot
    plt.figure(figsize=(15, 8))
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df = metrics_df.drop('support', axis=1)
    metrics_df.plot(kind='bar')
    plt.title('Per-class Metrics', fontsize=14, pad=20)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(['Precision', 'Recall', 'F1-score'], fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{base_output}_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Class Distribution
    plt.figure(figsize=(15, 8))
    class_counts = pd.Series(y_true).value_counts()
    class_counts.plot(kind='bar')
    plt.title('Class Distribution', fontsize=14, pad=20)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{base_output}_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Performance Summary
    plt.figure(figsize=(10, 6))
    plt.axis('off')
    summary_text = f"""
    Overall Performance:
    Accuracy: {report['accuracy']:.3f}
    Macro Avg F1: {report['macro avg']['f1-score']:.3f}
    Weighted Avg F1: {report['weighted avg']['f1-score']:.3f}
    
    Best Performing Classes:
    {', '.join(sorted(classes, key=lambda x: report[x]['f1-score'], reverse=True)[:3])}
    
    Most Challenging Classes:
    {', '.join(sorted(classes, key=lambda x: report[x]['f1-score'])[:3])}
    """
    plt.text(0.1, 0.5, summary_text, fontsize=14, va='center')
    plt.tight_layout()
    plt.savefig(f'{base_output}_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed metrics to CSV
    metrics_df.to_csv(f'{base_output}_metrics.csv')
    
    print(f"Performance visualizations saved as:")
    print(f"- {base_output}_confusion_matrix.png")
    print(f"- {base_output}_metrics.png")
    print(f"- {base_output}_distribution.png")
    print(f"- {base_output}_summary.png")
    print(f"Detailed metrics saved to {base_output}_metrics.csv")
    
    return report

def main():
    accuracy, all_labels, all_preds = test_model(0)

    # Generate performance report
    report = generate_performance_report(
        all_labels,
        all_preds,
        test_dataset.classes,
        options.output_file
    )
    
    print(f"Performance report saved to {options.output_file}")
    print(f"Detailed metrics saved to class_metrics.csv")

if __name__ == "__main__":
    main()