"""
PyTorch Dataset for financial time series
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict

from ..utils.config import get_config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class FinancialDataset(Dataset):
    """
    PyTorch Dataset for financial time series sequences
    """

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        tickers: Optional[List[str]] = None,
        timestamps: Optional[List[pd.Timestamp]] = None,
        transform=None
    ):
        """
        Initialize dataset

        Args:
            sequences: Array of shape (n_samples, sequence_length, n_features)
            targets: Array of shape (n_samples,) or (n_samples, n_targets)
            tickers: Optional list of ticker symbols for each sample
            timestamps: Optional list of timestamps for each sample
            transform: Optional transform to apply
        """
        self.sequences = torch.FloatTensor(sequences)

        # Handle both 1D and 2D targets
        if len(targets.shape) == 1:
            self.targets = torch.FloatTensor(targets).unsqueeze(1)
        else:
            self.targets = torch.FloatTensor(targets)

        self.tickers = tickers or []
        self.timestamps = timestamps or []
        self.transform = transform

        logger.info(f"Dataset initialized with {len(self)} samples, "
                   f"sequence shape: {self.sequences[0].shape}, "
                   f"target shape: {self.targets[0].shape}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get item by index

        Returns:
            Tuple of (sequence, target, metadata)
        """
        sequence = self.sequences[idx]
        target = self.targets[idx]

        if self.transform:
            sequence = self.transform(sequence)

        metadata = {
            'index': idx,
            'ticker': self.tickers[idx] if idx < len(self.tickers) else None,
            'timestamp': self.timestamps[idx] if idx < len(self.timestamps) else None,
        }

        return sequence, target, metadata

    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        return {
            'n_samples': len(self),
            'sequence_length': self.sequences.shape[1],
            'n_features': self.sequences.shape[2],
            'target_shape': self.targets.shape[1:],
            'sequences_mean': self.sequences.mean(dim=(0, 1)).numpy(),
            'sequences_std': self.sequences.std(dim=(0, 1)).numpy(),
            'targets_mean': self.targets.mean().item(),
            'targets_std': self.targets.std().item(),
        }


class PhysicsAwareDataset(FinancialDataset):
    """
    Extended dataset that includes additional information for physics-informed training
    """

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        tickers: Optional[List[str]] = None,
        timestamps: Optional[List[pd.Timestamp]] = None,
        prices: Optional[np.ndarray] = None,
        returns: Optional[np.ndarray] = None,
        volatilities: Optional[np.ndarray] = None,
        dt: float = 1.0,  # Time step in days
        transform=None
    ):
        """
        Initialize physics-aware dataset

        Args:
            sequences: Feature sequences
            targets: Target values
            tickers: Ticker symbols
            timestamps: Timestamps
            prices: Price sequences for GBM/Black-Scholes
            returns: Return sequences for statistical moments
            volatilities: Volatility sequences
            dt: Time step (default: 1 day)
            transform: Optional transform
        """
        super().__init__(sequences, targets, tickers, timestamps, transform)

        # Physics-specific data
        self.prices = torch.FloatTensor(prices) if prices is not None else None
        self.returns = torch.FloatTensor(returns) if returns is not None else None
        self.volatilities = torch.FloatTensor(volatilities) if volatilities is not None else None
        self.dt = dt

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Get item with physics information"""
        sequence, target, metadata = super().__getitem__(idx)

        # Add physics-specific data to metadata
        if self.prices is not None:
            metadata['prices'] = self.prices[idx]

        if self.returns is not None:
            metadata['returns'] = self.returns[idx]

        if self.volatilities is not None:
            metadata['volatilities'] = self.volatilities[idx]

        metadata['dt'] = self.dt

        return sequence, target, metadata


def collate_fn_with_metadata(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Custom collate function that handles metadata

    Args:
        batch: List of (sequence, target, metadata) tuples

    Returns:
        Tuple of (batched sequences, batched targets, batched metadata)
    """
    sequences = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])

    # Aggregate metadata
    metadata = {
        'indices': [item[2]['index'] for item in batch],
        'tickers': [item[2]['ticker'] for item in batch],
        'timestamps': [item[2]['timestamp'] for item in batch],
    }

    # Add physics-specific metadata if present
    if 'prices' in batch[0][2]:
        metadata['prices'] = torch.stack([item[2]['prices'] for item in batch])

    if 'returns' in batch[0][2]:
        metadata['returns'] = torch.stack([item[2]['returns'] for item in batch])

    if 'volatilities' in batch[0][2]:
        metadata['volatilities'] = torch.stack([item[2]['volatilities'] for item in batch])

    if 'dt' in batch[0][2]:
        metadata['dt'] = batch[0][2]['dt']

    return sequences, targets, metadata


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train/val/test datasets

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size (None = use config)
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    config = get_config()
    batch_size = batch_size or config.training.batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_with_metadata,
        drop_last=True,  # Drop last incomplete batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_with_metadata,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn_with_metadata,
        drop_last=False,
    )

    logger.info(f"Created DataLoaders: "
               f"Train={len(train_loader)} batches, "
               f"Val={len(val_loader)} batches, "
               f"Test={len(test_loader)} batches")

    return train_loader, val_loader, test_loader
