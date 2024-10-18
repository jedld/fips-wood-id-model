import os
import argparse

def prepare_dataset(base_path, class_labels_file):
    # Define paths
    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')

    # Create dataset, train, and test directories if they don't exist
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Read class labels from the file
    with open(class_labels_file, 'r') as file:
        class_labels = file.read().splitlines()

    # Create subdirectories for each class label in train and test directories
    for label in class_labels:
        label = label.replace(' ', '_')
        os.makedirs(os.path.join(train_path, label), exist_ok=True)
        os.makedirs(os.path.join(test_path, label), exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset directories.')
    parser.add_argument('base_path', type=str, help='Base path for the dataset')
    parser.add_argument('class_labels_file', type=str, help='Path to the class labels file')
    
    args = parser.parse_args()
    
    prepare_dataset(args.base_path, args.class_labels_file)