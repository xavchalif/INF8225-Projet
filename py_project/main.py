import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader

from dataset import PneumoniaDataset
from training import train_model

labels = ['opacity', 'normal']
img_size = 224
config = {
    'epochs': 10,
    'batch_size': 32,
    'lr': 1e-6,
    'betas': (0.9, 0.99),
    'clip': 2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'image_size': 224,
    'seed': 42,
    'log_every': 10,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(config['seed'])

print("Fetching Training data...")
train_dataset = PneumoniaDataset('../chest_xray/train/', labels)
val_dataset = PneumoniaDataset('../chest_xray/val/', labels)
test_dataset = PneumoniaDataset('../chest_xray/test/', labels)

print("Training data size:", len(train_dataset))
print("Validation data size:", len(val_dataset))
print("Testing data size:", len(test_dataset))

config['train_loader'] = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
config['val_loader'] = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
config['test_loader'] = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

model = models.resnet18(weights=None)

model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

model.fc = nn.Linear(model.fc.in_features, 1)

model = model.to(config['device'])

config['optimizer'] = optim.AdamW(model.parameters(), lr=config['lr'], betas=config['betas'])
config['criterion'] = nn.BCEWithLogitsLoss()

print(DEVICE)

train_model(model, config)
