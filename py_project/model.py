import torch
import torch.nn as nn
from torchinfo import summary
from torchvision import models


class ResNet(nn.Module):
    def __init__(self, config, labels):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.model.fc = nn.Sequential(
            nn.Dropout(p=config['dropout']),
            nn.Linear(self.model.fc.in_features, len(labels))
        )

    def forward(self, x):
        return self.model(x)

    def get_summary(self, config):
        model_summary = summary(
            self,
            input_size=(config['batch_size'], 1, 224, 224),
            dtypes=[torch.float],
            depth=3,
            verbose=0
        )

        nb_params = model_summary.total_params
        model_size = round(
            (model_summary.total_input + model_summary.total_param_bytes + model_summary.total_output_bytes) / 1e6,
            2)

        return nb_params, model_size
