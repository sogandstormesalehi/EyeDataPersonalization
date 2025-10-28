import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Computes attention weights over LSTM outputs."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim * 2, 1)  # bidirectional LSTM output

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden_dim*2)
        attn_weights = F.softmax(self.proj(lstm_out), dim=1)  # (B, T, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)   # (B, 2H)
        return context, attn_weights


class LSTMAttentionForecast(nn.Module):
    """
    Bidirectional LSTM + Attention model for gaze entropy forecasting.
    Predicts the next sequence of entropy values given past gaze dynamics.
    """
    def __init__(self, input_dim=5, hidden_dim=128, num_layers=2, pred_len=125, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.attn = AttentionLayer(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, pred_len)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)                # (B, T, 2H)
        context, attn = self.attn(lstm_out)       # (B, 2H)
        out = self.fc(context)                    # (B, pred_len)
        return out, attn
