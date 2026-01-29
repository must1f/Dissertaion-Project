"""
Baseline models: LSTM, GRU, Bidirectional LSTM
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from ..utils.logger import get_logger

logger = get_logger(__name__)


class LSTMModel(nn.Module):
    """
    LSTM baseline model for time series forecasting
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM model

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            output_dim: Output dimension (typically 1 for price prediction)
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Calculate output size considering bidirectionality
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            hidden: Optional hidden state tuple (h_0, c_0)

        Returns:
            Tuple of (output, (h_n, c_n))
        """
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)

        # Take the last output
        last_output = lstm_out[:, -1, :]

        # Fully connected layers
        output = self.fc(last_output)

        return output, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state

        Args:
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            Tuple of (h_0, c_0)
        """
        num_directions = 2 if self.bidirectional else 1

        h_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_dim
        ).to(device)

        c_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_dim
        ).to(device)

        return h_0, c_0


class GRUModel(nn.Module):
    """
    GRU baseline model for time series forecasting
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        Initialize GRU model

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            num_layers: Number of GRU layers
            output_dim: Output dimension
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional GRU
        """
        super(GRUModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional

        # GRU layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Calculate output size
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            hidden: Optional hidden state h_0

        Returns:
            Tuple of (output, h_n)
        """
        # GRU forward
        gru_out, hidden = self.gru(x, hidden)

        # Take the last output
        last_output = gru_out[:, -1, :]

        # Fully connected layers
        output = self.fc(last_output)

        return output, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state"""
        num_directions = 2 if self.bidirectional else 1

        h_0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.hidden_dim
        ).to(device)

        return h_0


class BiLSTMModel(LSTMModel):
    """
    Bidirectional LSTM model (convenience wrapper)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2
    ):
        """Initialize bidirectional LSTM"""
        super(BiLSTMModel, self).__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout,
            bidirectional=True
        )


class AttentionLSTM(nn.Module):
    """
    LSTM with attention mechanism for improved long-term dependencies
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2
    ):
        """
        Initialize Attention LSTM

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            output_dim: Output dimension
            dropout: Dropout probability
        """
        super(AttentionLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with attention

        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)
            hidden: Optional hidden state

        Returns:
            Tuple of (output, hidden)
        """
        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)

        # Attention weights
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Weighted sum of LSTM outputs
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_dim)

        # Final output
        output = self.fc(context)

        return output, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state"""
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return h_0, c_0
