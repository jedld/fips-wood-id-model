import os
import argparse
from PIL import Image
from collections import defaultdict

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
    return total_size

def format_size(size_bytes):
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def analyze_resolutions(folder_path):
    """Analyze image resolutions in a folder"""
    resolutions = defaultdict(int)
    total_images = 0
    
    for class_name in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_path):
            continue
            
        for filename in os.listdir(class_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                try:
                    with Image.open(os.path.join(class_path, filename)) as img:
                        resolution = img.size
                        resolutions[resolution] += 1
                        total_images += 1
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    
    return resolutions, total_images

def generate_report(dataset_folder):
    train_folder = os.path.join(dataset_folder, 'train')
    test_folder = os.path.join(dataset_folder, 'test')

    train_counts = {class_name: len(os.listdir(os.path.join(train_folder, class_name))) for class_name in os.listdir(train_folder)}
    test_counts = {class_name: len(os.listdir(os.path.join(test_folder, class_name))) for class_name in os.listdir(test_folder)}

    train_total = sum(train_counts.values())
    test_total = sum(test_counts.values())
    
    # Calculate total size of train and test datasets
    train_size = get_folder_size(train_folder)
    test_size = get_folder_size(test_folder)

    # Analyze resolutions
    train_resolutions, train_total_images = analyze_resolutions(train_folder)
    test_resolutions, test_total_images = analyze_resolutions(test_folder)

    report_lines = ["Training Dataset:\n"]
    for class_name in sorted(train_counts.keys()):
        report_lines.append(f"Class '{class_name}': {train_counts[class_name]} images\n")
    report_lines.append(f"Total images in training dataset: {train_total}\n")
    report_lines.append(f"Total size of training dataset: {format_size(train_size)}\n\n")

    report_lines.append("Training Dataset Resolutions:\n")
    for resolution, count in sorted(train_resolutions.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / train_total_images) * 100
        report_lines.append(f"Resolution {resolution}: {count} images ({percentage:.1f}%)\n")
    report_lines.append("\n")

    report_lines.append("Test Dataset:\n")
    for class_name in sorted(test_counts.keys()):
        report_lines.append(f"Class '{class_name}': {test_counts[class_name]} images\n")
    report_lines.append(f"Total images in test dataset: {test_total}\n")
    report_lines.append(f"Total size of test dataset: {format_size(test_size)}\n\n")

    report_lines.append("Test Dataset Resolutions:\n")
    for resolution, count in sorted(test_resolutions.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / test_total_images) * 100
        report_lines.append(f"Resolution {resolution}: {count} images ({percentage:.1f}%)\n")

    report_path = os.path.join(dataset_folder, 'dataset_report.txt')
    with open(report_path, 'w') as report_file:
        report_file.writelines(report_lines)

    print(f"Report generated: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a report on the number of images under each class for the training and test datasets.")
    parser.add_argument("dataset", type=str, default="data", help="Path to the dataset folder containing 'train' and 'test' subfolders.")
    args = parser.parse_args()

    generate_report(args.dataset)