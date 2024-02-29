import torch
import os
from PIL import Image
from tqdm import tqdm

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # Store (image path, class label) tuples
        self.load_images()

    def load_images(self):
        problematic_images = []  # Store paths of problematic images

        # Loop over the directories and subdirectories in the root directory
        for class_name in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            # Loop over the files in each subdirectory
            for filename in tqdm(os.listdir(class_dir), desc="Loading images", unit="image"):
                if filename.endswith('.jpg'):
                    img_path = os.path.join(class_dir, filename)
                    try:
                        # Open each image file
                        with Image.open(img_path) as img:
                            if self.transform:
                                img = self.transform(img)
                            # Add image path, class label, and class name to lists
                            label = self.get_label(class_name)
                            self.samples.append((img_path, label))
                    except Exception as e:
                        print(f"Skipping problematic image: {img_path}")
                        problematic_images.append(img_path)

        # Remove problematic images from the dataset
        for img_path in problematic_images:
            try:
                idx = [sample[0] for sample in self.samples].index(img_path)
                del self.samples[idx]
            except ValueError:
                print(f"Skipping problematic image: {img_path} not found in the dataset")

    def get_label(self, class_name):
        # Define a mapping from class name to label
        label_map = {
            'happy': 0,
            'embarrass': 1,
            'pain': 2,
            'anxiety': 3,
            'anger': 4,
            'normal': 5,
            'sad': 6
        }
        return label_map[class_name]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            with Image.open(img_path) as img:
                if self.transform:
                    img = self.transform(img)
                return img, label
        except Exception as e:
            print(f"Skipping problematic image: {img_path}")
            pass
