import os
import argparse
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

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

def generate_dataset_report(data_dir, output_file='dataset_report.png', bar_chart_only=False):
    """
    Generate a report of the dataset including class distribution visualization.
    
    Args:
        data_dir (str): Path to dataset directory
        output_file (str): Path to save the report visualization
        bar_chart_only (bool): If True, only show the bar chart
    """
    # Get all classes and their image counts
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    class_counts = {}
    
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        class_counts[class_name] = len(images)
    
    # Sort classes by count
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    class_names = [x[0] for x in sorted_classes]
    counts = [x[1] for x in sorted_classes]
    
    # Calculate statistics
    total_images = sum(counts)
    min_count = min(counts)
    max_count = max(counts)
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    
    # Create figure
    if bar_chart_only:
        plt.figure(figsize=(15, 8))
        # Single bar plot
        bars = plt.bar(range(len(class_names)), counts)
        plt.title('Class Distribution in Dataset')
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        # Add statistics text
        stats_text = f"""
        Dataset Statistics:
        Total Images: {total_images}
        Number of Classes: {len(classes)}
        Min Images per Class: {min_count}
        Max Images per Class: {max_count}
        Mean Images per Class: {mean_count:.1f}
        Std Dev of Images per Class: {std_count:.1f}
        """
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    else:
        # Original two-plot layout
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Bar plot of class distribution
        plt.subplot(2, 1, 1)
        bars = plt.bar(range(len(class_names)), counts)
        plt.title('Class Distribution in Dataset')
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        # Plot 2: Pie chart showing proportion of classes
        plt.subplot(2, 1, 2)
        plt.pie(counts, labels=class_names, autopct='%1.1f%%')
        plt.title('Proportion of Classes in Dataset')
        
        # Add statistics text
        stats_text = f"""
        Dataset Statistics:
        Total Images: {total_images}
        Number of Classes: {len(classes)}
        Min Images per Class: {min_count}
        Max Images per Class: {max_count}
        Mean Images per Class: {mean_count:.1f}
        Std Dev of Images per Class: {std_count:.1f}
        """
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dataset report generated and saved to {output_file}")
    print("\nDataset Statistics:")
    print(f"Total Images: {total_images}")
    print(f"Number of Classes: {len(classes)}")
    print(f"Min Images per Class: {min_count}")
    print(f"Max Images per Class: {max_count}")
    print(f"Mean Images per Class: {mean_count:.1f}")
    print(f"Std Dev of Images per Class: {std_count:.1f}")

def main():
    parser = argparse.ArgumentParser(description='Generate dataset analysis report')
    parser.add_argument('--data-dir', default='data/train', help='Path to dataset directory')
    parser.add_argument('--output-file', default='dataset_report.png', help='Path to save the report visualization')
    parser.add_argument('--bar-chart-only', action='store_true', help='Show only the bar chart')
    args = parser.parse_args()
    
    generate_dataset_report(args.data_dir, args.output_file, args.bar_chart_only)

if __name__ == '__main__':
    main()