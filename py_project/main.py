import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
import wandb
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchinfo import summary

from dataset import PneumoniaDataset
from training import train_model, eval_model, print_logs

labels = ['opacity', 'normal']
config = {
    'epochs': 10,
    'batch_size': 64,
    'lr': 0.00001,
    'betas': (0.9, 0.999),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'dropout': 0.1,
}

train_transforms = T.Compose([
    T.Resize(256),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomRotation(40),
    T.ToTensor(),
    T.Normalize(mean=[0.579], std=[0.164])
])

val_transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.579], std=[0.164])
])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(config['seed'])
wandb.login(key="25d52e5cb7f8dea06cf4dec651150fd6d91ecb71", verify=True)

print("-----------------------------------")
print("Fetching Training data...")
train_dataset = PneumoniaDataset('../chest_xray/train/', labels, train_transforms)
val_dataset = PneumoniaDataset('../chest_xray/val/', labels, val_transforms)
test_dataset = PneumoniaDataset('../chest_xray/test/', labels, val_transforms)

print("Training data size:", len(train_dataset))
print("Validation data size:", len(val_dataset))
print("Testing data size:", len(test_dataset))
print("-----------------------------------")

config['train_loader'] = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
config['val_loader'] = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
config['test_loader'] = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

model = models.resnet18(weights=None)

model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

model.fc = nn.Sequential(
    nn.Dropout(p=config['dropout']),
    nn.Linear(model.fc.in_features, len(labels))
)

model = model.to(config['device'])

config['optimizer'] = optim.AdamW(model.parameters(), lr=config['lr'], betas=config['betas'], weight_decay=1e-4)
config['scheduler'] = lr_scheduler.ExponentialLR(config['optimizer'], gamma=0.9)
config['criterion'] = nn.CrossEntropyLoss()

model_summary = summary(
    model,
    input_size=(config['batch_size'], 1, 224, 224),
    dtypes=[torch.float],
    depth=3,
    verbose=0
)

nb_params = model_summary.total_params
model_size = round(
    (model_summary.total_input + model_summary.total_param_bytes + model_summary.total_output_bytes) / 1e6, 2)

with wandb.init(
        config=config,
        entity="guipreg-polytechnique-montr-al",
        project='INF8225-PROJECT',
        group='torch',
        save_code=True,
):
    wandb.log({'nb_params': nb_params, 'model_size': model_size})
    trained_model = train_model(model, config)

print("Testing the model...")
test_logs = eval_model(trained_model, config, config['test_loader'], 'test')
print_logs('Test', test_logs)
wandb.log({**test_logs})

