import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

"""
    Helpers for the LSTM network.
"""

class SelectLastStep(nn.Module):
    def forward(self, x):
        lstm_out, _ = x
        return lstm_out[:, -1, :]
    
# Create a sliding window dataset for time series prediction
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


"""
    Helpers for data loading.
"""

def get_classification_dataset(dataset='cifar10', device='cpu', num_workers=1):

    if (dataset == 'cifar10'):
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False, generator=torch.Generator(device=device), num_workers=num_workers)

    elif (dataset == 'mnist'):

        transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False, generator=torch.Generator(device=device), num_workers=num_workers)

    else:
        raise ValueError(f"Invalid dataset {dataset} provided.")

    return test_loader

def get_regression_dataset(dataset='airline', sequence_length=3):
        
    if (dataset == 'airline'):
        df = pd.read_csv("./model_checkpoints/airline-passengers.csv")
        months, passengers = df.iloc[:, 0].values, df.iloc[:, 1].values.reshape(-1, 1)

        scaler = MinMaxScaler()
        data_normalized = scaler.fit_transform(passengers)

        X, y = create_sequences(data_normalized, sequence_length)
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()

        # although the train split used for training is 2/3, we return the full dataset here
        # makes for a nicer visualization later
        test_dataset = TimeSeriesDataset(X, y)
        data_loader = DataLoader(test_dataset, batch_size=int(len(X)), shuffle=False)

        xs = months

    else:
        raise ValueError(f"Invalid dataset {dataset} provided.")

    return data_loader, scaler, xs


"""
    Helpers for performing inference.
"""

def test_classifier(dataloader, model, device, compute_cm=False, log=False):
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    cm = None
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total
    if (log): print(f'Accuracy on test set: {acc:.2f}%')
    if (compute_cm): cm = confusion_matrix(all_labels, all_preds)
    return acc, cm

def test_lstm(dataloader, model, device, log=False):

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        mse = np.mean((all_preds - all_labels) ** 2)
        rmse = np.sqrt(mse)

    if (log): print(f'RMSE on test set: {rmse:.5f}%')

    return rmse

"""
    Miscellaneous helpers.
"""

def update_sd_by_idx(old_sd, new_sd):
    """
        Loads new_sd index-wise and in-place
    """

    # The loaded state dict has a slightly different format (due to WAGE activation quantization operations)
    # we exclude them here for simplicity by re-mapping the state dicts directly

    old_keys = list(old_sd.keys())
    new_keys = list(new_sd.keys())
    
    # Map weights element-wise based on index
    for k, new_key in enumerate(new_keys): new_sd[new_key] = old_sd[old_keys[k]]

def find_optimal_model_epoch(all_metrics, dataset):
    all_metrics = np.array(all_metrics)
    
    # Let's find the overall best model; this is the model that gets the best accuracy/RMSE across all models and epochs
    optimal_idx = np.argmin(all_metrics) if dataset == 'airline' else np.argmax(all_metrics)
    optimal_model, optimal_epoch = np.unravel_index(optimal_idx, all_metrics.shape)
    
    # Let's also find the best model across each independent run; the reported std is based on this
    optimal_metric = np.min(all_metrics, axis=1) if dataset == 'airline' else np.max(all_metrics, axis=1)

    return optimal_model, optimal_epoch, optimal_metric