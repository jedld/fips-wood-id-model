import torch
import numpy as np
from torchvision.utils import save_image
from gan_model import Generator
import os

# Hyperparameters
latent_dim = 100
img_size = 512
channels = 3
num_classes = 31  # Update this based on your dataset
img_shape = (channels, img_size, img_size)

# Load generator model
generator = Generator(num_classes, latent_dim, img_shape)
generator.load_state_dict(torch.load('generator_checkpoint.pth'))
generator.eval()

def generate_images(class_label, num_images=10):
    z = torch.randn(num_images, latent_dim)
    labels = torch.tensor([class_label] * num_images)
    gen_imgs = generator(z, labels)
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale images to [0, 1]
    if os.path.exists('generated_images') == False:
        os.mkdir('generated_images')
    for i in range(num_images):
        save_image(gen_imgs[i], f"generated_images/class_{class_label}_img_{i}.png")

if __name__ == "__main__":
    class_label = int(input("Enter the class label: "))
    generate_images(class_label)