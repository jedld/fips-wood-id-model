"""
Training of Conditional SA-GAN

Programmed by [Your Name]
* Date: Updated to support labels for conditional generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import random
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import gradient_penalty, save_checkpoint, load_checkpoint
from gan_model import Discriminator, Generator, initialize_weights  # Make sure your updated models are in model.py
from torch.cuda.amp import autocast, GradScaler
from torchsummary import summary
import argparse

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
CRITIC_LEARNING_RATE = 1e-4
GENERATOR_LEARNING_RATE = 1e-4
BATCH_SIZE = 8
IMAGE_SIZE = 512
CHANNELS_IMG = 3
Z_DIM = 100  # Adjusted to match the default in your updated Generator
NUM_EPOCHS = 10000
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

# Define transformations
transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(degrees=[0, 90, 180, 270]),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

# Load dataset with labels
dataset = datasets.ImageFolder(root="data/train", transform=transform)
num_classes = len(dataset.classes)  # Get number of classes from dataset

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# Initialize generator and discriminator
gen = Generator(nz=Z_DIM, num_classes=num_classes).to(device)
critic = Discriminator(num_classes=num_classes).to(device)
initialize_weights(gen)
initialize_weights(critic)


# Initialize optimizers
opt_gen = optim.Adam(gen.parameters(), lr=GENERATOR_LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=CRITIC_LEARNING_RATE, betas=(0.0, 0.9))

# Load models if available
if os.path.exists(f"best_generator_model.pth"):
    gen.load_state_dict(torch.load(f"best_generator_model.pth"))
    if os.path.exists(f"generator_opt.pth"):
        opt_gen.load_state_dict(torch.load(f"generator_opt.pth"))
else:
    print("Generator weights not found!")

if os.path.exists(f"best_discriminator_model.pth"):
    critic.load_state_dict(torch.load(f"best_discriminator_model.pth"))
    if os.path.exists(f"critic_opt.pth"):
        opt_critic.load_state_dict(torch.load(f"critic_opt.pth"))
else:
    print("Discriminator weights not found!")

start_epoch = 0

if os.path.exists(f"meta.pth"):
    meta = torch.load(f"meta.pth")
    start_epoch = meta["epoch"]

def save_images(images, labels, epoch, max_images=20, save_dir="generated_images"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Convert the images for visualization
    images = images.detach().cpu().numpy()[:max_images]  # Cap the number of images
    labels = labels.detach().cpu().numpy()[:max_images]
    images = (images + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
    images = np.transpose(images, (0, 2, 3, 1))  # Change to (batch, height, width, channel)

    num_images = len(images)

    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(4, 5, i + 1)
        plt.imshow(images[i])
        plt.title(f"Class: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle(f"Generated Images at Epoch {epoch}", fontsize=16)
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}.png"))
    plt.close()  # Close the plot to free up resources

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter(log_dir="runs/gan_training")

# For tensorboard plotting (optional)
fixed_noise = torch.randn(32, Z_DIM).to(device)
fixed_labels = torch.arange(0, num_classes).repeat(2)
fixed_labels = fixed_labels[:32].to(device)

parser = argparse.ArgumentParser()
parser.add_argument('--use_amp', type=bool, default=False, help='Whether to use Automatic Mixed Precision or not')
args = parser.parse_args()

use_amp = args.use_amp  # Set this to False if you don't want to use AMP

if use_amp:
    scaler = GradScaler()
else:
    scaler = None  # This will help avoid using the scaler if AMP is turned off

for epoch in range(start_epoch, NUM_EPOCHS):
    gen.train()
    critic.train()

    for batch_idx, (real, labels) in enumerate(tqdm(loader)):
        real = real.to(device)
        labels = labels.to(device).long()  # Convert labels to LongTensor
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            # Sample noise and labels for generator input
            noise = torch.randn(cur_batch_size, Z_DIM).to(device)
            fake_labels = torch.randint(0, num_classes, (cur_batch_size,), device=device).long()  # Convert labels to LongTensor
            fake = gen(noise, fake_labels)

            critic.zero_grad()

            # Use autocast for mixed-precision forward pass
            with autocast(enabled=use_amp):
                critic_real = critic(real, labels).reshape(-1)
                critic_fake = critic(fake.detach(), fake_labels).reshape(-1)
                gp = gradient_penalty(critic, real, fake.detach(), labels, device=device)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                )

            if use_amp:
                scaler.scale(loss_critic).backward(retain_graph=True)
                scaler.step(opt_critic)
                scaler.update()
            else:
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

        # Train Generator
        noise = torch.randn(cur_batch_size, Z_DIM).to(device)
        fake_labels = torch.randint(0, num_classes, (cur_batch_size,), device=device).long()  # Convert labels to LongTensor
        fake = gen(noise, fake_labels)

        gen.zero_grad()

        with autocast(enabled=use_amp):
            gen_fake = critic(fake, fake_labels).reshape(-1)
            loss_gen = -torch.mean(gen_fake)

        if use_amp:
            scaler.scale(loss_gen).backward()
            scaler.step(opt_gen)
            scaler.update()
        else:
            loss_gen.backward()
            opt_gen.step()

        if batch_idx % 100 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

    # Log losses to TensorBoard
    writer.add_scalar("Loss/Discriminator", loss_critic.item(), epoch)
    writer.add_scalar("Loss/Generator", loss_gen.item(), epoch)

    # Print losses occasionally and save generated images
    if epoch > 0:
        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
        )
        gen.eval()
        with torch.no_grad():
            # Generate fixed noise and labels for consistent visualization
            fake_images = gen(fixed_noise, fixed_labels)
            save_images(fake_images, fixed_labels, epoch, 20, f"generated_images")

            # Generate and log images for 3 random classes
            random_classes = torch.randint(0, num_classes, (3,), device=device)
            for cls in random_classes:
                noise = torch.randn(1, Z_DIM).to(device)
                label = torch.tensor([cls], device=device).long()  # Convert label to LongTensor
                fake_image = gen(noise, label)
                writer.add_image(f"Generated Images/Class_{cls}", fake_image[0], epoch)

        gen.train()

    # Save models and optimizer states
    torch.save(critic.state_dict(), f"best_discriminator_model.pth")
    torch.save(gen.state_dict(), f"best_generator_model.pth") 
    torch.save(opt_gen.state_dict(), f"generator_opt.pth")
    torch.save(opt_critic.state_dict(), f"critic_opt.pth")
    torch.save({ "epoch": epoch }, f"meta.pth")

# Close the TensorBoard writer
writer.close()
