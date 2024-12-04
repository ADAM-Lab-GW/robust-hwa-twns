import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_metric_vs_epoch(all_metrics, dataset, y_label = 'Test Accuracy (%)'):
    plt.figure()
    all_metrics = np.array(all_metrics)
    x = range(all_metrics.shape[1])
    y = np.average(all_metrics, axis=0)
    error = np.std(all_metrics, axis=0)
    
    plt.plot(x, y)
    plt.fill_between(x, y-error, y+error, alpha=0.2)
    
    plt.xlabel('Epoch (#)')
    plt.ylabel(y_label)

    plt.title(dataset)

def plot_weight_histograms(model, dataset=''):
    """
    Plot 3D histograms of weights layer by layer for a PyTorch model.

    Parameters:
        model (torch.nn.Module): The PyTorch model to visualize weights.
    """
    # Prepare data
    histograms = []
    layer_names = []

    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            layer_names.append(f"({name.split('.')[1]})")
            weights = param.detach().cpu().numpy().flatten()
            histograms.append(weights)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create histograms (for the 3d plot)
    bins = 30
    for idx, (layer_name, weights) in enumerate(zip(layer_names, histograms)):
        hist, bin_edges = np.histogram(weights, bins=bins, density=True)
        x_positions = (bin_edges[:-1] + bin_edges[1:]) / 2  # Bin centers
        y_positions = np.full_like(x_positions, idx)
        z_positions = np.zeros_like(x_positions)

        dx = (bin_edges[1] - bin_edges[0]) * np.ones_like(x_positions)
        dy = np.full_like(x_positions, 0.5)
        dz = hist

        # Plot bars for network parameters
        ax.bar3d(x_positions, y_positions, z_positions, dx, dy, dz, alpha=0.7)

    ax.set_xlabel('Weight Values')
    ax.set_ylabel('Layer Index')
    ax.set_zlabel('Frequency')
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, rotation=45, ha='right')
    ax.set_title(f'{dataset}: Ternary Weight Histograms by Layer')

def plot_airline_predictions(scaler, all_preds, all_labels, months, subset_ticks=20):
    # Rescale predictions back to original scale
    predictions = scaler.inverse_transform(all_preds)
    actuals = scaler.inverse_transform(all_labels)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(actuals, label="Actual data")
    plt.plot(predictions, label="Predictions")

    ax.set_xticks(range(len(months))[::subset_ticks])
    ax.set_xticklabels(months[::subset_ticks])

    ax.axvline(int(len(months) * 2/3), color='gray', linestyle='--', label='Train/Test Split')

    plt.xlabel("Time")
    plt.ylabel("Number of Passengers")
    plt.legend()