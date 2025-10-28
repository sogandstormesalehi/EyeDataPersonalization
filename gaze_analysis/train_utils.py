import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, test_loader, epochs=50, lr=1e-3, patience=5):
    """
    Generic training loop for LSTMAttentionForecast.
    Includes early stopping and plots training curves.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = np.inf
    patience_counter = 0
    history = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred, _ = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        history["train"].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                y_pred, _ = model(x)
                val_loss += criterion(y_pred, y).item()

        val_loss /= len(test_loader)
        history["val"].append(val_loss)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Early stopping logic
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_attention_lstm.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Plot learning curves
    plt.figure(figsize=(6,4))
    plt.plot(history["train"], label="Train")
    plt.plot(history["val"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    model.load_state_dict(torch.load("best_attention_lstm.pt"))
    return model, history
