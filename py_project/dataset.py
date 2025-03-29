import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class PneumoniaDataset(Dataset):
    def __init__(self, data_dir, labels):
        self.transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                                    T.Normalize(mean=[0.583], std=[0.141])])
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

        label = torch.tensor(label, dtype=torch.float).unsqueeze(0)
        return image, label
