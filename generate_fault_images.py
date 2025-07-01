import os
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import random
import shutil

def create_fault_images(source_dir, output_dir, n_images=100):
    """
    Create fault class images by applying various transformations to create blurry and invalid images.
    
    Args:
        source_dir (str): Directory containing source images to transform
        output_dir (str): Directory to save fault images
        n_images (int): Number of fault images to generate
    """
    # Create output directory
    fault_dir = os.path.join(output_dir, 'fault')
    os.makedirs(fault_dir, exist_ok=True)
    
    # Get all images from source directory
    all_images = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(root, file))
    
    # Define transformations
    def apply_blur(img):
        # Random Gaussian blur
        kernel_size = random.choice([5, 7, 9, 11])
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    def apply_motion_blur(img):
        # Motion blur
        size = random.randint(15, 30)
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel = kernel / size
        return cv2.filter2D(img, -1, kernel)
    
    def apply_noise(img):
        # Add random noise
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        return cv2.add(img, noise)
    
    def apply_rotation(img):
        # Random rotation
        angle = random.uniform(-45, 45)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        return cv2.warpAffine(img, M, (w, h))
    
    def apply_crop(img):
        # Random crop
        h, w = img.shape[:2]
        crop_size = random.randint(int(min(h, w) * 0.3), int(min(h, w) * 0.7))
        x = random.randint(0, w - crop_size)
        y = random.randint(0, h - crop_size)
        return img[y:y+crop_size, x:x+crop_size]
    
    def apply_darken(img):
        # Random darkening
        brightness = random.uniform(0.3, 0.7)
        return cv2.convertScaleAbs(img, alpha=brightness, beta=0)
    
    def create_solid_color_image():
        # Create a solid color image (black, white, or gray)
        color_type = random.choice(['black', 'white', 'gray'])
        if color_type == 'black':
            color = (0, 0, 0)
        elif color_type == 'white':
            color = (255, 255, 255)
        else:  # gray
            gray_value = random.randint(50, 200)
            color = (gray_value, gray_value, gray_value)
        
        # Create image with random size between 256x256 and 1024x1024
        size = random.randint(256, 1024)
        img = np.full((size, size, 3), color, dtype=np.uint8)
        return img
    
    # Generate fault images
    for i in range(n_images):
        # Randomly decide whether to create a solid color image or transform an existing one
        if random.random() < 0.2:  # 20% chance of creating a solid color image
            img = create_solid_color_image()
        else:
            # Randomly select source image
            src_img_path = random.choice(all_images)
            img = cv2.imread(src_img_path)
            
            # Apply random combination of transformations
            transforms = [
                apply_blur,
                apply_motion_blur,
                apply_noise,
                apply_rotation,
                apply_crop,
                apply_darken
            ]
            
            # Apply 2-4 random transformations
            n_transforms = random.randint(2, 4)
            selected_transforms = random.sample(transforms, n_transforms)
            
            for transform in selected_transforms:
                img = transform(img)
        
        # Save transformed image
        output_path = os.path.join(fault_dir, f'fault_{i:04d}.jpg')
        cv2.imwrite(output_path, img)
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1} fault images")

if __name__ == "__main__":
    # Create fault images for both train and test sets
    create_fault_images('data/train', 'data/train', n_images=50)  # 50 images for training
    create_fault_images('data/test', 'data/test', n_images=20)    # 20 images for testing 