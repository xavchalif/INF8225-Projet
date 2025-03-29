import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class PneumoniaDataset(Dataset):
    def __init__(self, data_dir, labels, transform=None):
        self.transform = transform
        self.data = []  # Store (image_path, label) tuples
        for label in labels:
            path = os.path.join(data_dir, label)
            class_num = labels.index(label)
            for img in os.listdir(path):
                img_path = os.path.join(path, img)
                self.data.append((img_path, class_num))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)
        return image, label
