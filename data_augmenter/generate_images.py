import torch
import numpy as np
from torchvision.utils import save_image
from gan_model import Generator
import os

# Hyperparameters
latent_dim = 100
img_size = 512
channels = 3
NUMBER_OF_IMAGES_PER_CLASS = 10
num_classes = 31  # Update this based on your dataset
img_shape = (channels, img_size, img_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load generator model
generator = Generator(nz=latent_dim, num_classes=num_classes).to(device)

# Load the model with map_location
model_path = 'best_generator_model.pth'
if torch.cuda.is_available():
    generator.load_state_dict(torch.load(model_path))
else:
    generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
generator.eval()

def generate_images(class_label, num_images=10):
    z = torch.randn(num_images, latent_dim).to(device)
    labels = torch.tensor([class_label] * num_images).to(device)
    gen_imgs = generator(z, labels)
    return gen_imgs

def save_generated_images(class_label, num_images=10, class_labels=None):
    gen_imgs = generate_images(class_label, num_images)
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale images to [0, 1]
    if os.path.exists('generated_images') == False:
        os.mkdir('generated_images')
    class_label_str = class_labels[class_label].strip()
    folder = f"generated_images/{class_label_str}"
    if os.path.exists(folder) == False:
        os.mkdir(folder)

    for i in range(len(gen_imgs)):

        padded_index = str(i).zfill(3)
        save_image(gen_imgs[i], f"{folder}/img_{padded_index}.png")

if __name__ == "__main__":
    # load class labels string to index mapping
    # load class_labels.txt

    with open('class_labels.txt', 'r') as f:
        class_labels = f.readlines()

    for i in range(num_classes):
        save_generated_images(i, class_labels=class_labels)