import os
import argparse

def count_images_in_folder(folder_path):
    class_counts = {}
    total_count = 0
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            class_folder = os.path.join(root, dir_name)
            image_count = len([f for f in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, f))])
            class_counts[dir_name] = image_count
            total_count += image_count
    return class_counts, total_count

def generate_report(dataset_folder):
    train_folder = os.path.join(dataset_folder, 'train')
    test_folder = os.path.join(dataset_folder, 'test')

    train_counts, train_total = count_images_in_folder(train_folder)
    test_counts, test_total = count_images_in_folder(test_folder)

    report_lines = ["Dataset Report\n"]
    report_lines.append("Training Dataset:\n")
    for class_name, count in train_counts.items():
        report_lines.append(f"Class '{class_name}': {count} images\n")
    report_lines.append(f"Total images in training dataset: {train_total}\n")

    report_lines.append("\nTest Dataset:\n")
    for class_name, count in test_counts.items():
        report_lines.append(f"Class '{class_name}': {count} images\n")
    report_lines.append(f"Total images in test dataset: {test_total}\n")

    report_path = os.path.join(dataset_folder, 'dataset_report.txt')
    with open(report_path, 'w') as report_file:
        report_file.writelines(report_lines)

    print(f"Report generated: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a report on the number of images under each class for the training and test datasets.")
    parser.add_argument("dataset_folder", type=str, default="data", help="Path to the dataset folder containing 'train' and 'test' subfolders.")
    args = parser.parse_args()

    generate_report(args.dataset_folder)