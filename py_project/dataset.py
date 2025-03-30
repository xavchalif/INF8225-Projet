import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class PneumoniaDataset(Dataset):
    def __init__(self, data_dir, labels, transforms):
        self.transform = transforms
        self.data = []
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


def create_dataloader(data_dir, labels, config, split):
    if split == 'train':
        transforms = T.Compose([
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomRotation(40),
            T.ToTensor(),
            T.Normalize(mean=[0.579], std=[0.164])
        ])
        dataset = PneumoniaDataset(data_dir, labels, transforms)
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    else:
        transforms = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.579], std=[0.164])
        ])
        dataset = PneumoniaDataset(data_dir, labels, transforms)
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)

    return dataloader
