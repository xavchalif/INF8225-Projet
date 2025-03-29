import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


def print_logs(dataset_type: str, logs: dict):
    desc = '\t'.join([f'{name}: {value:.4f}' for name, value in logs.items()])
    print(f'{dataset_type} -\t{desc}'.expandtabs(5))


def eval_model(model: nn.Module, dataloader: DataLoader, config: dict) -> dict:
    model.eval()  # Set model to evaluation mode
    device = config['device']
    criterion = config['criterion']
    logs = defaultdict(list)
    model.to(device)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images)
            loss = criterion(outputs, labels)
            logs['val_loss'].append(loss.cpu().item())

            preds = torch.sigmoid(outputs).round()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds)
    logs['val_f1_score'] = f1

    return {name: np.mean(values) for name, values in logs.items()}



def train_model(model: nn.Module, config: dict):
    train_loader, val_loader = config['train_loader'], config['val_loader']
    optimizer = config['optimizer']
    criterion = config['criterion']
    clip = config['clip']
    device = config['device']

    start_time = time.time()

    print(f'Starting training for {config["epochs"]} epochs on {device}.')
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()  # Start time for each epoch
        print(f'\nEpoch {epoch + 1}')

        model.train()
        train_loss = 0.0
        for batch_id, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        epoch_end_time = time.time()  # End time for the current epoch
        epoch_duration = epoch_end_time - epoch_start_time  # Calculate epoch duration
        print(f"Time taken: {epoch_duration:.2f} seconds")

        # Log metrics
        train_logs = {
            'train_loss': train_loss,
            # 'epoch_duration': epoch_duration
        }
        print_logs('Train', train_logs)

        val_logs = eval_model(model, val_loader, config)
        print_logs('Eval', val_logs)

        # Log everything to W&B
        # wandb.log({**train_logs, **val_logs})

    total_training_time = time.time() - start_time  # Total training time
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")
