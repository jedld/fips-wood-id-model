import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from pathlib import Path
import time
from datetime import timedelta
import datetime
import argparse
import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image

# ImageNet normalization stats
NORMALIZATION_STATS = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

class FaultDetectionModel(nn.Module):
    def __init__(self, pretrained=True):
        super(FaultDetectionModel, self).__init__()
        # Use ResNet18 as the base model
        self.model = models.resnet18(pretrained=pretrained)
                # Modify the final layer for binary classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.model(x)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))

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
        self.dataset = ImageFolder(root=root)
        self.transform = transform
        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = np.array(img)
        
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]
        else:
            img = cv2.resize(img, (512, 512))
            img = torch.from_numpy(img).float()
            if img.ndim == 2:
                img = img.unsqueeze(0)
            elif img.ndim == 3 and img.shape[0] != 3:
                img = img.permute(2, 0, 1)
        
        # Convert label to float tensor with shape [1] to match model output
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        
        return img, label

def plot_metrics(train_losses, test_losses, train_accs, test_accs, save_path='fault_detection_curves.png'):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    
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

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Train a fault detection model')
    parser.add_argument('--augmentation', default=True, help='Use augmentation')
    parser.add_argument('--pretrained', default=True, help='Use pretrained model')
    parser.add_argument('--plot', action='store_true', help='Plot training curves')
    parser.add_argument('--size', type=int, default=512, help='Input size')
    parser.add_argument('--name', default=None, help='Name of the model')
    parser.add_argument('--description', default="Fault Detection Model", help='Description of the model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--pin-memory', action='store_true', help='Use pinned memory for faster data transfer to GPU')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision training')
    parser.add_argument('--max-epochs', type=int, default=100, help='Maximum number of training epochs')
    args = parser.parse_args()

    # Generate model name with timestamp if not provided
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.name is None:
        args.name = f"fault_detection_{timestamp}"
    
    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Define checkpoint paths
    checkpoint_path = checkpoint_dir / f"{args.name}.pth"
    model_info_path = checkpoint_dir / f"{args.name}.txt"
    
    # Save model information
    with open(model_info_path, 'w') as f:
        f.write(f"Model Type: Fault Detection (ResNet18)\n")
        f.write(f"Input Size: {args.size}x{args.size}\n")
        f.write(f"Normalization Stats: {NORMALIZATION_STATS}\n")
        f.write(f"Created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Description: {args.description}\n")
        f.write(f"Pretrained: {args.pretrained}\n")
        f.write(f"Augmentation: {args.augmentation}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Mixed Precision: {args.amp}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True

    print(f"Using device: {device}")
    print(f"Input size: {args.size}x{args.size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Using {'pinned memory' if args.pin_memory else 'regular memory'}")
    print(f"Using {'mixed precision' if args.amp else 'full precision'} training")

    # Set random seed
    torch.manual_seed(1337)

    # Get transforms
    train_transform, test_transform = get_transforms(args)
    
    # Create datasets
    train_dataset = AlbumentationsDataset(root='fault_dataset/train', transform=train_transform)
    test_dataset = AlbumentationsDataset(root='fault_dataset/test', transform=test_transform)

    trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=args.pin_memory,
        persistent_workers=True
    )

    testloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=args.pin_memory,
        persistent_workers=True
    )

    # Initialize model
    model = FaultDetectionModel(pretrained=args.pretrained)
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler
    steps_per_epoch = len(trainloader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=args.max_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1e4
    )

    # Initialize mixed precision training
    scaler = GradScaler()

    # Initialize TensorBoard writer
    writer = SummaryWriter()

    # Initialize lists to store metrics
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    def test_model(epoch):
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        avg_loss = test_loss / len(testloader)
        
        print(f'Test [Epoch {epoch + 1}] Loss: {avg_loss:.3f}, Accuracy: {accuracy:.3f}')
        
        writer.add_scalar('Test/Loss', avg_loss, epoch)
        writer.add_scalar('Test/Accuracy', accuracy, epoch)
        
        test_losses.append(avg_loss)
        test_accs.append(accuracy)
        
        return accuracy

    best_accuracy = test_model(0)

    # Training loop
    for epoch in tqdm.tqdm(range(args.max_epochs)):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Add L2 regularization
                l2_lambda = 0.01
                l2_reg = torch.tensor(0., device=device)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                loss += l2_lambda * l2_reg
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 10 == 9:
                avg_loss = running_loss / 10
                accuracy = correct / total
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {avg_loss:.3f}, Accuracy: {accuracy:.3f}')
                writer.add_scalar('Train/Loss', avg_loss, epoch * len(trainloader) + i)
                writer.add_scalar('Train/Accuracy', accuracy, epoch * len(trainloader) + i)
                running_loss = 0.0
                correct = 0
                total = 0
        
        # Evaluate model
        accuracy = test_model(epoch)
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"New best accuracy: {accuracy:.3f}")
            model.save_weights(checkpoint_path)
            
            with open(model_info_path, 'a') as f:
                f.write(f"Best Accuracy: {accuracy:.3f} at epoch {epoch}\n")
    
    print(f'Finished Training. Best accuracy: {best_accuracy:.3f}')
    
    # Calculate and print total training time
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total training time: {timedelta(seconds=int(total_time))}')
    
    if args.plot:
        plot_metrics(train_losses, test_losses, train_accs, test_accs)
    
    writer.close()

if __name__ == "__main__":
    main() 