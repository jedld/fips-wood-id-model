#!/usr/bin/env python3

import os
import argparse
from pathlib import Path

def add_prefix_to_images(directory: str, prefix: str):
    """
    Add a prefix to all PNG and JPEG files in the specified directory and its subdirectories.
    
    Args:
        directory (str): Path to the directory containing images
        prefix (str): Prefix to add to the filenames
    """
    # Convert directory to Path object
    dir_path = Path(directory)
    
    # Check if directory exists
    if not dir_path.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    # Walk through directory and subdirectories
    for root, _, files in os.walk(directory):
        # Filter for PNG and JPEG files
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            continue
        
        # Rename files
        for filename in image_files:
            old_path = Path(root) / filename
            new_filename = f"{prefix}{filename}"
            new_path = Path(root) / new_filename
            
            try:
                old_path.rename(new_path)
                print(f"Renamed: {old_path} -> {new_path}")
            except OSError as e:
                print(f"Error renaming {old_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Add prefix to PNG and JPEG files in a directory and its subdirectories')
    parser.add_argument('directory', help='Directory containing the images')
    parser.add_argument('prefix', help='Prefix to add to the filenames')
    
    args = parser.parse_args()
    
    add_prefix_to_images(args.directory, args.prefix)

if __name__ == '__main__':
    main() 