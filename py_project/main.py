import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from dataset import PneumoniaDataset

labels = ['PNEUMONIA', 'NORMAL']
img_size = 224
config = {
    'epochs': 10,
    'batch_size': 32,
    'lr': 1e-3,
    'betas': (0.9, 0.99),
    'clip': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'image_size': 150,
    'dropout': 0.1,
    'seed': 42,
    'log_every': 10,
}

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                       T.Normalize(mean=[0.583], std=[0.141])])

print("Fetching Training data...")
train_dataset = PneumoniaDataset('../chest_xray/train/', labels, transform=transform)
val_dataset = PneumoniaDataset('../chest_xray/val/', labels, transform=transform)
test_dataset = PneumoniaDataset('../chest_xray/test/', labels, transform=transform)


print("Training data size:", len(train_dataset))
print("Validation data size:", len(val_dataset))
print("Testing data size:", test_dataset.data[0].shape, test_dataset.data[1].shape)

#
# x_train = x_train.reshape(-1, img_size, img_size, 1)
# y_train = np.array(y_train)
#
# x_val = x_val.reshape(-1, img_size, img_size, 1)
# y_val = np.array(y_val)
#
# x_test = x_test.reshape(-1, img_size, img_size, 1)
# y_test = np.array(y_test)


# config['train_loader'] = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
# config['val_loader'] = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
