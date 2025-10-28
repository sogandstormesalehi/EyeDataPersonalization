import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, loader, n_batches=10):
    """
    Evaluate a trained model on test data using R² and Pearson correlation.
    Also plots sample predictions vs ground truth sequences.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model.
    loader : DataLoader
        DataLoader for test set.
    n_batches : int
        Number of batches to evaluate for quick inspection.

    Returns
    -------
    (r2, corr) : tuple
        Coefficient of determination and Pearson correlation.
    """
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            y_pred, _ = model(x)
            preds.append(y_pred.cpu().numpy())
            trues.append(y.cpu().numpy())
            if i >= n_batches:
                break

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    y_true_flat = trues.flatten()
    y_pred_flat = preds.flatten()

    r2 = r2_score(y_true_flat, y_pred_flat)
    corr, _ = pearsonr(y_true_flat, y_pred_flat)

    print(f"R² score: {r2:.4f}")
    print(f"Pearson correlation: {corr:.4f}")

    idx = np.random.choice(len(trues), size=min(3, len(trues)), replace=False)
    plt.figure(figsize=(12, 4))
    for i, ix in enumerate(idx, start=1):
        plt.subplot(1, 3, i)
        plt.plot(trues[ix], label='True', linewidth=1.5)
        plt.plot(preds[ix], '--', label='Predicted', linewidth=1.2)
        plt.title(f'Sample {ix}')
        plt.xlabel('Time (samples)')
        plt.ylabel('Rolling Entropy')
        if i == 1:
            plt.legend()
    plt.suptitle("Example Forecasts (Next 5s)")
    plt.tight_layout()
    plt.show()

    return r2, corr
