import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
    
class RNN(nn.Module):
    """
    """

    def __init__(
        self,
        C_in,
        output_size,
        C_out=32,
        kernel_size=5,
        hidden_size=32,
        n_layers=2,
        dropout=0.1,
        ):
        """
        """

        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=C_in,
                out_channels=C_out,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                dilation=1,
            ),
            nn.ReLU()
        )

        # RNN encoder
        self.rnn = nn.LSTM(
            input_size=C_out,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)

        # MLP regressor
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

        return
    
    def forward(self, X):
        """
        """

        #
        X_tp = X.transpose(1, 2) # N x U (units) x T
        X_conv = self.cnn(X_tp)  # N x F (new features) x T
        X_conv = X_conv.transpose(1, 2) # N x T x F
        out, (h_n, c_n) = self.rnn(X_conv) # out is N x T x H (number of hidden states)
        h_summary = out.mean(dim=1) # N x H
        h_summary = self.dropout(h_summary) # N x H
        y = self.head(h_summary) # N x output size

        return y
    
class MLP(nn.Module):
    """
    Simple feedforward MLP: input -> hidden layers -> output.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layer_sizes=(128,),
        activation=nn.ReLU,
        dropout: float = 0.1,
        ):
        super().__init__()

        layers = []
        in_dim = input_size

        for h_dim in hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(activation())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_size))
        self.seq = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.seq(X)