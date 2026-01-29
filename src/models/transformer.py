"""
Transformer model for time series forecasting
"""

import math
import torch
import torch.nn as nn
from typing import Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer to capture temporal information
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding

        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input

        Args:
            x: Input tensor (batch_size, sequence_length, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer model for time series forecasting
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        output_dim: int = 1,
        max_len: int = 5000
    ):
        """
        Initialize Transformer model

        Args:
            input_dim: Number of input features
            d_model: Model dimension (must be divisible by nhead)
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            output_dim: Output dimension
            max_len: Maximum sequence length for positional encoding
        """
        super(TransformerModel, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead

        # Ensure d_model is divisible by nhead
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
        self.input_embedding.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)
            src_mask: Optional attention mask
            src_key_padding_mask: Optional padding mask

        Returns:
            Output tensor (batch_size, output_dim)
        """
        # Input embedding
        x = self.input_embedding(x) * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        transformer_out = self.transformer_encoder(
            x,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )

        # Use the last output for prediction
        last_output = transformer_out[:, -1, :]

        # Final output
        output = self.fc(last_output)

        return output

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        """
        Generate a square mask for the sequence (for autoregressive prediction)

        Args:
            sz: Size of the mask
            device: Device to create mask on

        Returns:
            Attention mask
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


class TransformerEncoderDecoder(nn.Module):
    """
    Full Transformer with encoder-decoder architecture for sequence-to-sequence prediction
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        max_len: int = 5000
    ):
        """
        Initialize encoder-decoder Transformer

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super(TransformerEncoderDecoder, self).__init__()

        self.d_model = d_model

        # Input embeddings
        self.src_embedding = nn.Linear(input_dim, d_model)
        self.tgt_embedding = nn.Linear(output_dim, d_model)

        # Positional encodings
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, output_dim)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            src: Source sequence (batch_size, src_seq_len, input_dim)
            tgt: Target sequence (batch_size, tgt_seq_len, output_dim)
            Various masks for attention

        Returns:
            Output predictions (batch_size, tgt_seq_len, output_dim)
        """
        # Embeddings
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        # Positional encoding
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        # Transformer
        output = self.transformer(
            src=src,
            tgt=tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        # Output projection
        output = self.output_projection(output)

        return output
