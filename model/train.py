import torch
from image_resnet_transfer_classifier import ImageResNetTransferClassifier
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
from PIL import Image
from torch.autograd import Variable
import data
import torch.nn as nn
import cv2
import numpy as np
from collections import Counter
from torch.utils.data.sampler import Sampler
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler

from mobilenet import MobileNetV2TransferClassifier
from stn_model import MinimalCNN
from Resnet18 import ResNet18
from efficientnet import EfficientNetTransferClassifier
import torchvision
from torchvision import datasets
import tqdm


import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from custom_transforms import ResizeWithPadding
from torch.optim import lr_scheduler
import argparse
import time
from datetime import timedelta
import datetime
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, SWALR
import albumentations as A
from albumentations.pytorch import ToTensorV2

# imagenet normalization
NORMALIZATION_STATS = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# custom normalization
# NORMALIZATION_STATS = [0.5184, 0.3767, 0.3015], [0.2343, 0.2134, 0.1974]

def plot_metrics(train_losses, test_losses, train_accs, test_accs, save_path='training_curves.png'):
    """Plot training and test metrics over epochs."""
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

class BalancedBatchSampler(Sampler):
    """Samples batches with equal number of samples from each class."""
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.class_indices = {}
        
        # Group indices by class
        for idx, (_, label) in enumerate(dataset.samples):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        # Calculate samples per class per batch
        self.samples_per_class = batch_size // len(self.class_indices)
        if self.samples_per_class == 0:
            self.samples_per_class = 1
            print(f"Warning: Batch size {batch_size} is smaller than number of classes {len(self.class_indices)}. "
                  f"Using 1 sample per class.")
        
        # Calculate total number of batches
        self.num_batches = min(len(indices) // self.samples_per_class 
                             for indices in self.class_indices.values())
        
        # Calculate class weights for sampling
        class_counts = {label: len(indices) for label, indices in self.class_indices.items()}
        total_samples = sum(class_counts.values())
        self.class_weights = {label: total_samples / (len(class_counts) * count) 
                            for label, count in class_counts.items()}
        
        print("Class distribution in dataset:")
        for label, count in class_counts.items():
            print(f"Class {dataset.classes[label]}: {count} samples")
        print(f"Using {self.samples_per_class} samples per class per batch")

    def __iter__(self):
        # Create batches
        for _ in range(self.num_batches):
            batch_indices = []
            for class_label in self.class_indices:
                # Sample indices for this class
                indices = np.random.choice(
                    self.class_indices[class_label],
                    size=self.samples_per_class,
                    replace=len(self.class_indices[class_label]) < self.samples_per_class
                )
                batch_indices.extend(indices)
            
            # Shuffle the batch
            np.random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return self.num_batches

class EnhancedDataAugmentation:
    """Enhanced data augmentation for underrepresented classes."""
    def __init__(self, base_transforms, class_weights):
        self.base_transforms = base_transforms
        self.class_weights = class_weights
        
    def __call__(self, img, label):
        # Apply base transforms
        img = self.base_transforms(img)
        
        # For underrepresented classes, apply additional augmentations
        if self.class_weights[label] > 1.5:  # If class is underrepresented
            if np.random.random() < 0.5:
                # Additional rotation
                angle = np.random.uniform(-30, 30)
                img = F.rotate(img, angle)
            
            if np.random.random() < 0.5:
                # Additional color jitter
                img = transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                )(img)
            
            if np.random.random() < 0.5:
                # Additional random erasing
                img = transforms.RandomErasing(
                    p=1.0,
                    scale=(0.02, 0.1),
                    ratio=(0.3, 3.3),
                    value='random'
                )(img)
        
        return img

class OversamplingSampler(Sampler):
    """Samples elements with oversampling for minority classes."""
    def __init__(self, dataset, batch_size, oversample_factor=2.0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.oversample_factor = oversample_factor
        
        # Get class distribution
        self.class_indices = {}
        for idx, (_, label) in enumerate(dataset.samples):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        # Calculate class weights for oversampling
        class_counts = {label: len(indices) for label, indices in self.class_indices.items()}
        max_count = max(class_counts.values())
        self.class_weights = {label: min(max_count / count, 5.0) for label, count in class_counts.items()}
        
        # Calculate total samples after oversampling
        self.total_samples = int(sum(len(indices) * self.class_weights[label] 
                                   for label, indices in self.class_indices.items()))
        
        print("Class distribution before oversampling:")
        for label, count in class_counts.items():
            print(f"Class {dataset.classes[label]}: {count} samples")
        
        print("\nOversampling factors (capped at 5x):")
        for label, weight in self.class_weights.items():
            print(f"Class {dataset.classes[label]}: {weight:.2f}x")
        
        print(f"\nTotal samples after oversampling: {self.total_samples}")

    def __iter__(self):
        # Create indices with oversampling
        indices = []
        for label, label_indices in self.class_indices.items():
            # Calculate how many times to repeat this class's indices
            repeat_count = int(len(label_indices) * self.class_weights[label])
            # Add repeated indices
            indices.extend(np.random.choice(label_indices, size=repeat_count, replace=True))
        
        # Shuffle all indices
        np.random.shuffle(indices)
        
        # Yield individual indices
        for idx in indices:
            yield idx

    def __len__(self):
        return self.total_samples

def get_transforms(args):
    if args.augmentation:
        # Training transforms with albumentations
        train_transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                rotate=(-45, 45),
                translate_percent=(-0.0625, 0.0625),
                scale=(0.9, 1.1),
                p=0.5
            ),
            A.OneOf([
                A.ElasticTransform(
                    alpha=120,
                    sigma=120 * 0.05,
                    p=0.5
                ),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(
                    distort_limit=1,
                    p=0.5
                ),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=2, p=0.5),
                A.Sharpen(p=0.5),
                A.Emboss(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ], p=0.3),
            A.OneOf([
                A.RandomResizedCrop(
                    size=(args.size, args.size),
                    scale=(0.8, 1.0),
                    p=0.5
                ),
                A.RandomSizedCrop(
                    min_max_height=(int(args.size*0.8), args.size),
                    size=(args.size, args.size),
                    p=0.5
                ),
            ], p=0.3),
            A.Resize(args.size, args.size),
            A.Normalize(mean=NORMALIZATION_STATS[0], std=NORMALIZATION_STATS[1]),
            ToTensorV2(),
        ])
    else:
        train_transform = A.Compose([
            A.Resize(args.size, args.size),
            A.Normalize(mean=NORMALIZATION_STATS[0], std=NORMALIZATION_STATS[1]),
            ToTensorV2(),
        ])

    # Test transforms
    test_transform = A.Compose([
        A.Resize(args.size, args.size),
        A.Normalize(mean=NORMALIZATION_STATS[0], std=NORMALIZATION_STATS[1]),
        ToTensorV2(),
    ])

    return train_transform, test_transform

class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.dataset = datasets.ImageFolder(root=root)
        self.transform = transform
        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Convert PIL image to numpy array
        img = np.array(img)
        
        # Always apply transforms to ensure consistent size
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]
        else:
            # If no transform provided, at least ensure consistent size
            img = cv2.resize(img, (512, 512))
            img = torch.from_numpy(img).float()
            if img.ndim == 2:
                img = img.unsqueeze(0)
            elif img.ndim == 3 and img.shape[0] != 3:
                img = img.permute(2, 0, 1)
        
        return img, label

def main():
    # Start timing
    start_time = time.time()

    # Add command line arguments
    parser = argparse.ArgumentParser(description='Train a wood identification model')
    parser.add_argument('--grayscale', default=False, help='Use grayscale images instead of color')
    parser.add_argument('--augmentation', default=True, help='Use augmentation')
    parser.add_argument('--pretrained', default=True, help='Use pretrained model')
    parser.add_argument('--model', default='efficientnet', help='Model to use')
    parser.add_argument('--clahe', action='store_true', help='Use CLAHE preprocessing')
    parser.add_argument('--plot', action='store_true', help='Plot training curves')
    parser.add_argument('--size', type=int, default=512, help='Input size (512 or 1024)', metavar='SIZE')
    parser.add_argument('--name', default=None, help='Name of the model')
    parser.add_argument('--description', default="Wood Identification Model", help='Description of the model')
    parser.add_argument('--balanced', action='store_true', help='Use oversampling to balance the dataset')
    parser.add_argument('--oversample-factor', type=float, default=2.0, help='Factor to oversample minority classes')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--pin-memory', action='store_true', help='Use pinned memory for faster data transfer to GPU')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision training')
    parser.add_argument('--max-epochs', type=int, default=200, help='Maximum number of training epochs')
    args = parser.parse_args()

    # Generate model name with timestamp if not provided
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.name is None:
        args.name = f"{args.model}_{timestamp}"
    
    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Define checkpoint paths
    checkpoint_path = checkpoint_dir / f"{args.name}.pth"
    model_info_path = checkpoint_dir / f"{args.name}.txt"
    
    # Save model information
    with open(model_info_path, 'w') as f:
        f.write(f"Model Type: {args.model}\n")
        f.write(f"Input Size: {args.size}x{args.size}\n")
        f.write(f"Grayscale: {args.grayscale}\n")
        f.write(f"Normalization Stats: {NORMALIZATION_STATS}\n")
        f.write(f"Created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Description: {args.description}\n")
        f.write(f"Pretrained: {args.pretrained}\n")
        f.write(f"Augmentation: {args.augmentation}\n")
        f.write(f"Balanced Dataset: {args.balanced}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Mixed Precision: {args.amp}\n")

    batch_size = args.batch_size
    divfac = 4
    best_epoch = 0
    MAX_EPOCHS = args.max_epochs
    NUM_WORKERS = 16
    NUM_WORKERS_TEST = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enable cuDNN benchmarking for faster training
    if torch.cuda.is_available():
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True

    print(device)
    print(f"Using {'grayscale' if args.grayscale else 'color'} images")
    print(f"Input size: {args.size}x{args.size}")
    print(f"Batch size: {batch_size}")
    print(f"Using {'pinned memory' if args.pin_memory else 'regular memory'}")
    print(f"Using {'mixed precision' if args.amp else 'full precision'} training")

    # set random seed
    torch.manual_seed(1337)

    print("threads %d" % (torch.get_num_threads()))

    # Get transforms
    train_transform, test_transform = get_transforms(args)
    
    # Create datasets with albumentations
    train_dataset = AlbumentationsDataset(root='data/train', transform=train_transform)
    test_dataset = AlbumentationsDataset(root='data/test', transform=test_transform)

    if args.balanced:
        # Create oversampling sampler
        oversample_sampler = OversamplingSampler(
            train_dataset, 
            batch_size,
            oversample_factor=args.oversample_factor
        )
        
        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=oversample_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=args.pin_memory,
            persistent_workers=True
        )
    else:
        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=args.pin_memory,
            persistent_workers=True
        )

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=NUM_WORKERS_TEST,
        pin_memory=args.pin_memory,
        persistent_workers=True
    )

    total_classes = len(train_dataset.classes)
    print('Total classes %s' % (total_classes ))

    if args.model == 'minimalcnn':
        classifier = MinimalCNN(total_classes)
    elif args.model == 'mobilenet':
        classifier = MobileNetV2TransferClassifier(num_classes=total_classes, pretrained=args.pretrained)
    elif args.model == 'resnet18' and args.pretrained:
        classifier = ImageResNetTransferClassifier(num_classes=total_classes)
    elif args.model == 'resnet18':
        classifier = ResNet18(num_classes=total_classes, pretrained=args.pretrained, dropout_rate=0.2)
    elif args.model == 'efficientnet':
        classifier = EfficientNetTransferClassifier(num_classes=total_classes, pretrained=args.pretrained)
    else:
        raise ValueError(f"Unsupported model: {args.model}")


    if (train_dataset.classes != test_dataset.classes):
        print('Error test and train do not have the same classes')

    print(train_dataset.classes)

    PATH = "./checkpoint.pth"

    if Path(PATH).exists():
        print('loading weights')
        classifier.load_weights(Path(PATH))

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    print(' '.join('%5s' % train_dataset.classes[labels[j].item()] for j in range(min(batch_size, len(labels)))))

    with open('class_labels.txt', 'w') as the_file:
        for c in train_dataset.classes:
            the_file.write(c + '\n')

    classifier.eval()

    classifier = classifier.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = AdamW(classifier.parameters(), lr=0.001, weight_decay=0.01)
    
    # Initialize learning rate scheduler
    steps_per_epoch = len(trainloader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=MAX_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,  # 30% of training for warmup
        div_factor=25,  # initial_lr = max_lr/25
        final_div_factor=1e4  # final_lr = initial_lr/1e4
    )

    # Initialize SWA model
    swa_model = AveragedModel(classifier)
    swa_start = int(MAX_EPOCHS * 0.75)  # Start SWA at 75% of training
    swa_scheduler = SWALR(optimizer, swa_lr=0.0001)

    # Initialize mixed precision training
    scaler = GradScaler()

    # Initialize TensorBoard writer
    writer = SummaryWriter()

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
    inputs, classes = next(iter(trainloader))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    # Log images to TensorBoard
    writer.add_image('Train Images', out)

    # Print model summary
    summary(classifier, input_size=(3, args.size, args.size))

    # Initialize lists to store metrics
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

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
                    actual = test_dataset.classes[topk.indices.cpu().numpy()[0]]
                    all_labels.append(expected)
                    all_preds.append(actual)
                    if  expected == actual:
                        success+=1
                    else:
                        failure+=1

        accuracy = success / (success + failure)
        print('Test [%d] loss: %.3f Success: %d Failure: %d Accuracy: %.3f Total: %d' %
                    (epoch + 1, test_loss / total_items, success, failure, accuracy, success + failure))

        # Log test metrics to TensorBoard
        writer.add_scalar('Test/Loss', test_loss / total_items, epoch)
        writer.add_scalar('Test/Accuracy', accuracy, epoch)

        # Log test images to TensorBoard
        out = torchvision.utils.make_grid(inputs)
        writer.add_image('Test Images', out, epoch)

        # Store metrics for plotting
        test_losses.append(test_loss / total_items)
        test_accs.append(accuracy)

        return accuracy

    best_accuracy = test_model(0)

    # Training loop
    for epoch in tqdm.tqdm(range(MAX_EPOCHS)):
        running_loss = 0.0
        classifier.train()
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = classifier(inputs)
                loss = criterion(outputs, labels)
                
                # Add L2 regularization
                l2_lambda = 0.01
                l2_reg = torch.tensor(0., device=device)
                for param in classifier.parameters():
                    l2_reg += torch.norm(param)
                loss += l2_lambda * l2_reg
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            
            # Optimizer step with scaling
            scaler.step(optimizer)
            scaler.update()
            
            # Update learning rate
            scheduler.step()
            
            # Update SWA model
            if epoch >= swa_start:
                swa_model.update_parameters(classifier)
                swa_scheduler.step()
            
            running_loss += loss.item()
            
            if i % 10 == 9:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}')
                writer.add_scalar('Train/Loss', running_loss / 10, epoch * len(trainloader) + i)
                running_loss = 0.0
        
        # Evaluate model
        accuracy = test_model(epoch)
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            print(f"New best accuracy: {accuracy:.3f}")
            classifier.save_weights(checkpoint_path)
            
            # Update model info with best accuracy
            with open(model_info_path, 'a') as f:
                f.write(f"Best Accuracy: {accuracy:.3f} at epoch {epoch}\n")
            
            # Save SWA model if we're past the start epoch
            if epoch >= swa_start:
                torch.save(swa_model.state_dict(), f"{checkpoint_path}.swa")
    
    # Update batch normalization statistics for the SWA model
    if epoch >= swa_start:
        # Move SWA model to the same device as the classifier
        swa_model = swa_model.to(device)
        # Update batch normalization statistics
        swa_model.train()  # Set to training mode for BN update
        with torch.no_grad():
            for inputs, _ in trainloader:
                inputs = inputs.to(device)
                swa_model(inputs)
        torch.save(swa_model.state_dict(), f"{checkpoint_path}.swa.final")
    
    print(f'Finished Training. Best accuracy: {best_accuracy:.3f}')
    
    # Calculate and print total training time
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total training time: {timedelta(seconds=int(total_time))}')
    
    writer.close()

if __name__ == "__main__":
    main()