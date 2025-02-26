import torch
import torch.nn as nn


class CryptoLSTM(nn.Module):
    """
    LSTM model for cryptocurrency trend prediction.
    """
    def __init__(self, input_size=10, hidden_size=32, num_layers=2, dropout=0.3, output_size=3):
        super(CryptoLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)  # [long, neutral, short]
        )

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last timestep output
        return self.fc(x)
