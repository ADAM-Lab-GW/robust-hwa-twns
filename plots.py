import numpy as np
import matplotlib.pyplot as plt

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