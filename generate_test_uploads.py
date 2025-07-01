#!/usr/bin/env python3
"""
Script to generate test upload data for demonstrating the statistics functionality.
This creates sample image files in the upload directories.
"""

import os
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_test_image(filename, text="Test Image", size=(512, 512)):
    """Create a test image with text"""
    # Create a random colored background
    bg_color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
    img = Image.new('RGB', size, bg_color)
    
    # Add some random shapes
    draw = ImageDraw.Draw(img)
    
    # Draw some random rectangles
    for _ in range(5):
        x1 = random.randint(0, size[0]//2)
        y1 = random.randint(0, size[1]//2)
        x2 = random.randint(size[0]//2, size[0])
        y2 = random.randint(size[1]//2, size[1])
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.rectangle([x1, y1, x2, y2], fill=color)
    
    # Add text
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None
    
    # Calculate text position (center)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    # Draw text with outline
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    return img

def generate_test_uploads():
    """Generate test upload data"""
    
    # Get available classes
    try:
        with open('class_labels.txt', 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print("class_labels.txt not found, using sample classes")
        classes = ['acacia_auriculiformis', 'acacia_mangium', 'eucalyptus_camaldulensis', 
                  'falcataria_falcata', 'tectona_grandis']
    
    upload_folder = 'app/tmp/uploads'
    
    # Create upload folder if it doesn't exist
    os.makedirs(upload_folder, exist_ok=True)
    
    # Generate random number of uploads for each class
    total_uploads = 0
    
    for class_name in classes:
        # Random number of uploads (1-10) for each class
        num_uploads = random.randint(1, 10)
        
        # Create class directory
        class_dir = os.path.join(upload_folder, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        print(f"Generating {num_uploads} uploads for {class_name}...")
        
        for i in range(num_uploads):
            # Create test image
            img = create_test_image(
                f"test_{i+1}.jpg",
                f"{class_name.replace('_', ' ').title()}\nTest {i+1}",
                (512, 512)
            )
            
            # Save image
            filename = f"test_upload_{i+1:03d}.jpg"
            filepath = os.path.join(class_dir, filename)
            img.save(filepath, 'JPEG', quality=85)
            
            total_uploads += 1
    
    print(f"\n✅ Generated {total_uploads} test uploads across {len(classes)} classes")
    print(f"📁 Test data saved to: {upload_folder}")
    print("\nYou can now run the Flask app to see the statistics!")

if __name__ == "__main__":
    print("🌳 Wood ID Upload Statistics Test Data Generator")
    print("=" * 50)
    generate_test_uploads() 