import os
import argparse

def generate_report(dataset_folder):
    train_folder = os.path.join(dataset_folder, 'train')
    test_folder = os.path.join(dataset_folder, 'test')

    train_counts = {class_name: len(os.listdir(os.path.join(train_folder, class_name))) for class_name in os.listdir(train_folder)}
    test_counts = {class_name: len(os.listdir(os.path.join(test_folder, class_name))) for class_name in os.listdir(test_folder)}

    train_total = sum(train_counts.values())
    test_total = sum(test_counts.values())

    report_lines = ["Training Dataset:\n"]
    for class_name in sorted(train_counts.keys()):
        report_lines.append(f"Class '{class_name}': {train_counts[class_name]} images\n")
    report_lines.append(f"Total images in training dataset: {train_total}\n\n")

    report_lines.append("Test Dataset:\n")
    for class_name in sorted(test_counts.keys()):
        report_lines.append(f"Class '{class_name}': {test_counts[class_name]} images\n")
    report_lines.append(f"Total images in test dataset: {test_total}\n")

    report_path = os.path.join(dataset_folder, 'dataset_report.txt')
    with open(report_path, 'w') as report_file:
        report_file.writelines(report_lines)

    print(f"Report generated: {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a report on the number of images under each class for the training and test datasets.")
    parser.add_argument("dataset", type=str, default="data", help="Path to the dataset folder containing 'train' and 'test' subfolders.")
    args = parser.parse_args()

    generate_report(args.dataset)