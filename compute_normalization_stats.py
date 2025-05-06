import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

def process_image(img_path):
    """Process a single image and return its statistics"""
    try:
        img = Image.open(img_path)
        img_array = np.array(img)
        
        # Convert to float32 for better precision
        img_array = img_array.astype(np.float32) / 255.0
        
        # Reshape to (H*W, C) for easier computation
        h, w, c = img_array.shape
        img_array = img_array.reshape(-1, c)
        
        # Calculate statistics for this image
        sum_pixels = np.sum(img_array, axis=0)
        sum_squared_pixels = np.sum(img_array ** 2, axis=0)
        total_pixels = h * w
        
        return sum_pixels, sum_squared_pixels, total_pixels
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return np.zeros(3), np.zeros(3), 0

def compute_normalization_stats(dataset_folder):
    """
    Compute mean and standard deviation for each channel across training images only.
    Returns: (mean, std) as tuples of (R, G, B) values
    """
    # Initialize accumulators
    sum_pixels = np.zeros(3)
    sum_squared_pixels = np.zeros(3)
    total_pixels = 0
    total_images = 0

    # Get all image paths from training set only
    image_paths = []
    train_folder = os.path.join(dataset_folder, 'train')
    print("Collecting training images...")
    
    for class_name in os.listdir(train_folder):
        class_folder = os.path.join(train_folder, class_name)
        image_paths.extend([
            os.path.join(class_folder, f) 
            for f in os.listdir(class_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    if not image_paths:
        raise ValueError("No training images found in the dataset")

    # Process images in parallel
    num_processes = min(cpu_count(), len(image_paths))
    print(f"Using {num_processes} processes")
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_image, image_paths),
            total=len(image_paths),
            desc="Processing training images"
        ))

    # Aggregate results
    for sum_p, sum_sq, pixels in results:
        sum_pixels += sum_p
        sum_squared_pixels += sum_sq
        total_pixels += pixels
        total_images += 1

    # Compute mean and std
    mean = sum_pixels / total_pixels
    std = np.sqrt((sum_squared_pixels / total_pixels) - (mean ** 2))

    return mean, std, total_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute normalization statistics for a dataset using training images only")
    parser.add_argument("dataset", type=str, help="Path to the dataset folder containing 'train' and 'test' subfolders")
    args = parser.parse_args()

    print("Computing normalization statistics from training set only...")
    mean, std, total_images = compute_normalization_stats(args.dataset)
    
    print(f"\nProcessed {total_images} training images")
    
    # Format for easy copy-paste
    mean_str = f"[{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]"
    std_str = f"[{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]"
    
    print("\nCopy-paste ready values:")
    print(f"mean = {mean_str}")
    print(f"std = {std_str}")
    
    # Save to file in a format ready for copy-paste
    stats_path = os.path.join(args.dataset, 'normalization_stats.txt')
    with open(stats_path, 'w') as f:
        f.write("# Copy-paste these values into your transforms.Normalize()\n")
        f.write("# These statistics were computed using training images only\n")
        f.write(f"mean = {mean_str}\n")
        f.write(f"std = {std_str}\n")
        f.write("\n# Example usage:\n")
        f.write("transforms.Normalize(mean=mean, std=std)\n")
    
    print(f"\nStatistics saved to: {stats_path}") 