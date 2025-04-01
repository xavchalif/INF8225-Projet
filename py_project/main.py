import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import wandb
from dataset import create_dataloader
from model import Model
from training import train_model, eval_model, print_logs

labels = ['opacity', 'normal']
config = {
    'epochs': 5,
    'batch_size': 64,
    'lr': 0.00001,
    'betas': (0.9, 0.999),
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'dropout': 0.1,
    'log': 'online'  # 'online' for logging / 'offline' for testing
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(config['seed'])
wandb.login(key="25d52e5cb7f8dea06cf4dec651150fd6d91ecb71", verify=True)

config['train_loader'] = create_dataloader('../chest_xray/train/', labels, config, 'train')
config['val_loader'] = create_dataloader('../chest_xray/val/', labels, config, 'val')
config['test_loader'] = create_dataloader('../chest_xray/test/', labels, config, 'test')

model = Model(config, len(labels))
nb_params, model_size = model.get_summary(config)
resnet = model.resnet

resnet = resnet.to(config['device'])

config['optimizer'] = optim.AdamW(resnet.parameters(), lr=config['lr'], betas=config['betas'], weight_decay=1e-4)
config['scheduler'] = lr_scheduler.ExponentialLR(config['optimizer'], gamma=0.9)
config['criterion'] = nn.CrossEntropyLoss()

with wandb.init(
        mode=config['log'],
        config=config,
        entity="guipreg-polytechnique-montr-al",
        project='INF8225-PROJECT',
        group='torch',
        save_code=True,
):
    wandb.log({'nb_params': nb_params, 'model_size': model_size})
    trained_model = train_model(resnet, config)
    print("\nTesting the model...")
    test_logs = eval_model(trained_model, config, config['test_loader'], 'test')
    print_logs('Test', test_logs)
    wandb.log({**test_logs})
