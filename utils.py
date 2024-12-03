import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix

class SelectLastStep(nn.Module):
    def forward(self, x):
        lstm_out, _ = x
        return lstm_out[:, -1, :]
    
def get_test_loader(dataset='cifar10', device='cpu'):

    if (dataset == 'cifar10'):
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False, generator=torch.Generator(device=device), num_workers=1)

    elif (dataset == 'mnist'):
        return None

    return test_loader

def test_network(dataloader, model, device, compute_cm=False, log=False):
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    cm = None
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs = torch.flatten(inputs, start_dim=1)
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

def update_sd_by_idx(old_sd, new_sd):
    """
        Loads new_sd index-wise and in-place
    """
    # the loaded state dict has a slightly different format (due to WAGE operations)
    # we exclude them here for simplicity by re-mapping the state dicts directly

    # Get old and new keys
    old_keys = list(old_sd.keys())
    new_keys = list(new_sd.keys())
    
    # Map weights element-wise based on index
    for k, new_key in enumerate(new_keys):
      new_sd[new_key] = old_sd[old_keys[k]]