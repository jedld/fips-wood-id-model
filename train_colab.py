import torch
from torchvision.transforms import transforms
from PIL import Image
import torch.nn as nn
import cv2
import numpy as np
import torchvision
from torchvision import datasets
import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torch.optim import lr_scheduler
import argparse
import os
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Custom transform for resizing with padding
class ResizeWithPadding:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Create new image with padding
        new_img = Image.new('RGB', (self.size, self.size), (0, 0, 0))
        new_img.paste(img, ((self.size - new_w) // 2, (self.size - new_h) // 2))
        return new_img

# Normalization stats
NORMALIZATION_STATS = [0.5184, 0.3767, 0.3015], [0.2343, 0.2134, 0.1974]

def main():
    # Add command line arguments
    parser = argparse.ArgumentParser(description='Train a wood identification model')
    parser.add_argument('--grayscale', default=False, help='Use grayscale images instead of color')
    parser.add_argument('--pretrained', default=True, help='Use pretrained model')
    parser.add_argument('--model', default='mobilenet', help='Model to use')
    parser.add_argument('--clahe', action='store_true', help='Use CLAHE preprocessing')
    parser.add_argument('--data_dir', default='/content/drive/MyDrive/wood_dataset', help='Path to dataset directory')
    args = parser.parse_args()

    batch_size = 32
    divfac = 4
    best_epoch = 0
    MAX_EPOCHS = 100
    NUM_WORKERS = 2  # Reduced for Colab
    NUM_WORKERS_TEST = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using {'grayscale' if args.grayscale else 'color'} images")

    # set random seed
    torch.manual_seed(1337)

    # Define base transformations
    base_transforms = [
        ResizeWithPadding(512),
        transforms.RandomRotation(270),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(*NORMALIZATION_STATS),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random')
    ]

    # Add CLAHE transformation if requested
    if args.clahe:
        print("Using CLAHE preprocessing")
        def clahe_transform(img):
            img = np.array(img)
            if len(img.shape) == 2:  # Grayscale
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img = clahe.apply(img)
            else:  # Color
                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                lab = cv2.merge((l,a,b))
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return Image.fromarray(img)
        
        base_transforms.insert(0, transforms.Lambda(clahe_transform))

    # Add grayscale transformation if requested
    if args.grayscale:
        base_transforms.insert(0, transforms.Grayscale(num_output_channels=3))

    xfm3 = transforms.Compose(base_transforms)

    # Define test transformations
    test_transforms = [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(*NORMALIZATION_STATS)
    ]

    # Add CLAHE transformation for test if requested
    if args.clahe:
        test_transforms.insert(0, transforms.Lambda(clahe_transform))

    # Add grayscale transformation for test if requested
    if args.grayscale:
        test_transforms.insert(0, transforms.Grayscale(num_output_channels=3))

    xfm_test2 = transforms.Compose(test_transforms)

    # Load datasets
    train_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=xfm3)
    test_dataset = datasets.ImageFolder(root=os.path.join(args.data_dir, 'test'), transform=xfm_test2)

    fused_trainset = torch.utils.data.ConcatDataset([train_dataset])
    fused_testset = torch.utils.data.ConcatDataset([test_dataset])

    trainloader = torch.utils.data.DataLoader(fused_trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=NUM_WORKERS)
    testloader = torch.utils.data.DataLoader(fused_testset, batch_size=batch_size, num_workers=NUM_WORKERS_TEST)

    total_classes = len(train_dataset.classes)
    print('Total classes %s' % (total_classes))

    # Initialize model
    if args.model == 'mobilenet':
        from mobilenet import MobileNetV2TransferClassifier
        classifier = MobileNetV2TransferClassifier(num_classes=total_classes, pretrained=args.pretrained)
    elif args.model == 'resnet18':
        from Resnet18 import ResNet18
        classifier = ResNet18(num_classes=total_classes, pretrained=args.pretrained)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    if (train_dataset.classes != test_dataset.classes):
        print('Error test and train do not have the same classes')

    print(train_dataset.classes)

    # Save class labels
    with open('class_labels.txt', 'w') as the_file:
        for c in train_dataset.classes:
            the_file.write(c + '\n')

    classifier = classifier.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(classifier.parameters(), lr=0.0001, weight_decay=0.0005)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter()

    def test_model(epoch):
        test_loss = 0.0
        total_items = 0
        success = 0
        failure = 0
        classifier.eval()
        all_labels = []
        all_preds = []
        for _, data in enumerate(testloader):
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
                    all_labels.append(expected)
                    all_preds.append(actual)
                    if expected == actual:
                        success += 1
                    else:
                        failure += 1

        accuracy = success / (success + failure)
        print('Test [%d] loss: %.3f Success: %d Failure: %d Accuracy: %.3f Total: %d' %
                    (epoch + 1, test_loss / total_items, success, failure, accuracy, success + failure))

        writer.add_scalar('Test/Loss', test_loss / total_items, epoch)
        writer.add_scalar('Test/Accuracy', accuracy, epoch)

        out = torchvision.utils.make_grid(inputs)
        writer.add_image('Test Images', out, epoch)

        return accuracy

    best_accuracy = test_model(0)

    # Training loop
    for epoch in tqdm.tqdm(range(MAX_EPOCHS)):
        running_loss = 0.0
        classifier.train()
        success = 0
        failure = 0
        for i, row_data in tqdm.tqdm(enumerate(trainloader, 0)):
            inputs, labels = row_data[0].to(device), row_data[1].to(device)

            optimizer.zero_grad()
            outputs = classifier(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print('[%d, %5d]  loss: %.3f best: %.3f@%d ' %
                    (epoch + 1, i + 1, running_loss / 10, best_accuracy, best_epoch))
            writer.add_scalar('Train/Loss', running_loss / 10, epoch * len(trainloader) + i)
            running_loss = 0.0

        accuracy = test_model(epoch)
        if (accuracy > best_accuracy):
            best_accuracy = accuracy
            best_epoch = epoch
            print("best accuracy %f" % (accuracy))
            # Save model to Google Drive
            save_path = os.path.join('/content/drive/MyDrive/wood_model', 'checkpoint.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            classifier.save_weights(save_path)
            
            if args.grayscale:
                with open(os.path.join('/content/drive/MyDrive/wood_model', 'model_grayscale.txt'), 'w') as the_file:
                    the_file.write('%.3f@%d' % (accuracy, epoch))
                    the_file.write('\n')
                    the_file.write(str(NORMALIZATION_STATS))
            else:
                with open(os.path.join('/content/drive/MyDrive/wood_model', 'model.txt'), 'w') as the_file:
                    the_file.write('%.3f@%d' % (accuracy, epoch))
                    the_file.write('\n')
                    the_file.write(str(NORMALIZATION_STATS))

    print('Finished Training. Best accuracy %.3f' % (best_accuracy))
    writer.close()

if __name__ == "__main__":
    main() 