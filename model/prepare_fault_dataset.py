import os
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm

def copy_valid_images(source_dir, target_dir, class_name):
    """
    Copy valid images from source directory to target directory.
    
    Args:
        source_dir (str): Source directory containing valid images
        target_dir (str): Target directory to copy images to
        class_name (str): Name of the class (will be used as subdirectory)
    """
    # Create target directory if it doesn't exist
    target_path = Path(target_dir) / class_name
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files from source directory
    source_path = Path(source_dir)
    image_files = list(source_path.glob('**/*.jpg')) + list(source_path.glob('**/*.png'))
    
    # Copy files with progress bar
    for img_file in tqdm(image_files, desc=f"Copying {class_name} images"):
        # Create target filename
        target_file = target_path / img_file.name
        
        # Copy file if it doesn't exist in target
        if not target_file.exists():
            shutil.copy2(img_file, target_file)

def main():
    parser = argparse.ArgumentParser(description='Prepare fault detection dataset by copying valid images')
    parser.add_argument('--source-train', default='data/train', help='Source training data directory')
    parser.add_argument('--source-test', default='data/test', help='Source test data directory')
    parser.add_argument('--target-train', default='fault_dataset/train', help='Target training data directory')
    parser.add_argument('--target-test', default='fault_dataset/test', help='Target test data directory')
    args = parser.parse_args()

    # Create target directories if they don't exist
    Path(args.target_train).mkdir(parents=True, exist_ok=True)
    Path(args.target_test).mkdir(parents=True, exist_ok=True)

    print("Copying training images...")
    # Copy valid images from each class in training set
    for class_dir in Path(args.source_train).iterdir():
        if class_dir.is_dir():
            print(f"\nProcessing class: {class_dir.name}")
            copy_valid_images(
                class_dir,
                args.target_train,
                'valid'
            )

    print("\nCopying test images...")
    # Copy valid images from each class in test set
    for class_dir in Path(args.source_test).iterdir():
        if class_dir.is_dir():
            print(f"\nProcessing class: {class_dir.name}")
            copy_valid_images(
                class_dir,
                args.target_test,
                'valid'
            )

    print("\nDataset preparation completed!")
    
    # Print statistics
    train_valid_count = len(list(Path(args.target_train).glob('valid/*')))
    test_valid_count = len(list(Path(args.target_test).glob('valid/*')))
    
    print(f"\nDataset statistics:")
    print(f"Training valid images: {train_valid_count}")
    print(f"Test valid images: {test_valid_count}")

if __name__ == "__main__":
    main() 