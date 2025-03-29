import random
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


# Function to display 3 images on the same line
def display_images(images, predicted_labels, actual_labels):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns
    for i, ax in enumerate(axes):
        image = images[i].permute(1, 2, 0).cpu().numpy()  # Convert from Tensor (C, H, W) to (H, W, C)
        image = np.clip(image, 0, 1)  # Ensure values are between 0 and 1 for visualization
        ax.imshow(image)
        pred_label = "Pneumonia" if predicted_labels[i].item() == 0 else "Normal"
        actual_label = "Pneumonia" if actual_labels[i].item() == 0 else "Normal"
        ax.set_title(f'Pred: {pred_label}, Actual: {actual_label}')
        ax.axis('off')
    print(f"\nPredictions examples on validation set:")
    plt.show()


def print_logs(dataset_type: str, logs: dict):
    desc = '\t'.join([f'{name}: {value:.4f}' for name, value in logs.items()])
    print(f'{dataset_type} -\t{desc}'.expandtabs(5))


def eval_model(model: nn.Module, dataloader: DataLoader, config: dict) -> dict:
    device = config['device']
    logs = defaultdict(list)
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            logs['val_loss'].append(loss.cpu().item())

            # Collect predictions and labels for F1 calculation
            preds = torch.sigmoid(outputs).round()  # Sigmoid and round to get binary predictions
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate F1 score, etc.
    f1 = f1_score(all_labels, all_preds)
    logs['val_f1_score'] = f1

    return {name: np.mean(values) for name, values in logs.items()}


def train_model(model: nn.Module, config: dict):
    train_loader, val_loader = config['train_loader'], config['val_loader']
    optimizer = config['optimizer']
    clip = config['clip']
    device = config['device']

    start_time = time.time()  # Start time for total training

    print(f'Starting training for {config["epochs"]} epochs on {device}.')
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()  # Start time for each epoch
        print(f'\nEpoch {epoch + 1}')

        model.train()
        logs = defaultdict(list)
        correct = 0
        total = 0
        false_positive = 0
        false_negative = 0
        images_to_display = []

        # Training loop
        for batch_id, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device).unsqueeze(1).float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            logs['train_loss'].append(loss.cpu().item())

        epoch_end_time = time.time()  # End time for the current epoch
        epoch_duration = epoch_end_time - epoch_start_time  # Calculate epoch duration
        print(f"Time taken: {epoch_duration:.2f} seconds")

        # Now, collect 3 random images from the validation dataset to display
        model.eval()  # Set model to evaluation mode

        # Log metrics
        train_logs = {
            'train_loss': loss,
            'epoch_duration': epoch_duration
        }
        print_logs('Train', train_logs)

        val_logs = eval_model(model, val_loader, config)
        print_logs('Eval', val_logs)

        # Log everything to W&B
        wandb.log({**train_logs, **val_logs})

        with torch.no_grad():
            # Choose 3 random images from the validation set
            random_indices = random.sample(range(len(val_loader.dataset)), 3)
            images_batch = []
            predicted_labels = []
            actual_labels = []

            for idx in random_indices:
                image, label = val_loader.dataset[idx]
                image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
                output = model(image)
                pred = torch.sigmoid(output).round()

                images_batch.append(image[0])  # Remove batch dimension for display
                predicted_labels.append(pred[0])
                actual_labels.append(label)

            # Display 3 random images with predicted and actual labels
            display_images(images_batch, predicted_labels, actual_labels)

    total_training_time = time.time() - start_time  # Total training time
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")
