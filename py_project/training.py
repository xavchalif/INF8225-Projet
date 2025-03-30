import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb


def print_logs(dataset_type: str, logs: dict):
    desc = '\t'.join([f'{name}: {value:.4f}' for name, value in logs.items()])
    print(f'{dataset_type} -\t{desc}'.expandtabs(5))


def eval_model(model: nn.Module, config: dict, dataloader: DataLoader, split: str) -> dict:
    model.eval()
    device = config['device']
    criterion = config['criterion']
    model.to(device)
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(dataloader.dataset)
        epoch_accuracy = 100 * correct / total

        if split == 'train':
            logs = {
                'train_loss': epoch_val_loss,
                'epoch_duration': 0.0
            }
        elif split == 'val':
            logs = {
                'val_loss': epoch_val_loss,
                'val_accuracy': epoch_accuracy
            }
        else:
            logs = {
                'test_loss': epoch_val_loss,
                'test_accuracy': epoch_accuracy
            }

    return logs


def train_model(model: nn.Module, config: dict):
    train_loader = config['train_loader']
    optimizer = config['optimizer']
    criterion = config['criterion']
    scheduler = config['scheduler']
    epochs = config['epochs']
    device = config['device']
    print(f'Starting training for {epochs} epochs on {device}.')
    print(f'\nEpoch 0')
    train_logs = eval_model(model, config, config['train_loader'], 'train')
    print_logs('Train', train_logs)
    val_logs = eval_model(model, config, config['val_loader'], 'val')
    print_logs('Eval', val_logs)

    wandb.log({**train_logs, **val_logs})

    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f'\nEpoch {epoch + 1}')
        model.train()
        train_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            wandb.log({f'train_loss': loss.item()})

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        scheduler.step()
        train_logs = {
            'epoch_duration': epoch_duration
        }
        print_logs('Train', train_logs)

        val_logs = eval_model(model, config, config['val_loader'], 'val')
        print_logs('Eval', val_logs)

        wandb.log({**train_logs, **val_logs})

    return model
