import os
import shutil
import random
from pathlib import Path
import argparse

def create_human_eval_dataset(test_dir='data/test', output_dir='human_eval', n_samples=30):
    """
    Create a human evaluation dataset by randomly sampling from test set.
    
    Args:
        test_dir (str): Path to test dataset directory
        output_dir (str): Path to output directory for human evaluation
        n_samples (int): Number of samples to include
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all classes from test directory
    classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    
    # Calculate samples per class
    samples_per_class = n_samples // len(classes)
    remaining_samples = n_samples % len(classes)
    
    # Create mapping file
    mapping_file = os.path.join(output_dir, 'image_class_mapping.txt')
    with open(mapping_file, 'w') as f:
        f.write("filename,class\n")  # Header
        
        # Keep track of image index
        image_index = 1
        
        # Sample images from each class
        for class_name in classes:
            class_dir = os.path.join(test_dir, class_name)
            images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            # Calculate number of samples for this class
            n_class_samples = samples_per_class + (1 if remaining_samples > 0 else 0)
            remaining_samples -= 1
            
            # Randomly sample images
            selected_images = random.sample(images, min(n_class_samples, len(images)))
            
            # Copy and rename images
            for img in selected_images:
                # Generate sequential filename
                new_filename = f"{image_index}.jpg"
                
                # Copy image to output directory
                src_path = os.path.join(class_dir, img)
                dst_path = os.path.join(output_dir, new_filename)
                shutil.copy2(src_path, dst_path)
                
                # Write mapping to file
                f.write(f"{new_filename},{class_name}\n")
                
                # Increment image index
                image_index += 1
    
    print(f"Created human evaluation dataset with {n_samples} images")
    print(f"Mapping file created at: {mapping_file}")
    print(f"Images copied to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Create human evaluation dataset')
    parser.add_argument('--test-dir', default='data/test', help='Path to test dataset directory')
    parser.add_argument('--output-dir', default='human_eval', help='Path to output directory')
    parser.add_argument('--n-samples', type=int, default=30, help='Number of samples to include')
    args = parser.parse_args()
    
    create_human_eval_dataset(args.test_dir, args.output_dir, args.n_samples)

if __name__ == '__main__':
    main() 