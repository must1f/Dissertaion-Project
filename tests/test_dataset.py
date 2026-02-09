"""
Comprehensive tests for PyTorch dataset classes
"""

import pytest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    FinancialDataset,
    PhysicsAwareDataset,
    collate_fn_with_metadata,
    create_dataloaders
)


class TestFinancialDataset:
    """Test FinancialDataset class"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for dataset"""
        np.random.seed(42)
        n_samples = 100
        seq_len = 60
        n_features = 10
        
        sequences = np.random.randn(n_samples, seq_len, n_features)
        targets = np.random.randn(n_samples)
        tickers = [f'TICKER_{i % 5}' for i in range(n_samples)]
        timestamps = [pd.Timestamp('2020-01-01') + pd.Timedelta(days=i) for i in range(n_samples)]
        
        return sequences, targets, tickers, timestamps

    def test_dataset_initialization(self, sample_data):
        """Test dataset initialization"""
        sequences, targets, tickers, timestamps = sample_data
        
        dataset = FinancialDataset(sequences, targets, tickers, timestamps)
        
        assert len(dataset) == len(sequences)
        assert dataset.sequences.shape == (100, 60, 10)
        assert dataset.targets.shape == (100, 1)  # Should be unsqueezed
        assert len(dataset.tickers) == 100
        assert len(dataset.timestamps) == 100

    def test_dataset_initialization_2d_targets(self, sample_data):
        """Test dataset with 2D targets"""
        sequences, _, tickers, timestamps = sample_data
        targets_2d = np.random.randn(100, 3)  # Multiple output targets
        
        dataset = FinancialDataset(sequences, targets_2d, tickers, timestamps)
        
        assert dataset.targets.shape == (100, 3)

    def test_dataset_initialization_without_metadata(self, sample_data):
        """Test dataset without tickers/timestamps"""
        sequences, targets, _, _ = sample_data
        
        dataset = FinancialDataset(sequences, targets)
        
        assert len(dataset) == 100
        assert len(dataset.tickers) == 0
        assert len(dataset.timestamps) == 0

    def test_getitem(self, sample_data):
        """Test __getitem__ method"""
        sequences, targets, tickers, timestamps = sample_data
        
        dataset = FinancialDataset(sequences, targets, tickers, timestamps)
        
        sequence, target, metadata = dataset[0]
        
        assert isinstance(sequence, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert isinstance(metadata, dict)
        
        assert sequence.shape == (60, 10)
        assert target.shape == (1,)
        
        assert 'index' in metadata
        assert 'ticker' in metadata
        assert 'timestamp' in metadata
        
        assert metadata['index'] == 0
        assert metadata['ticker'] == 'TICKER_0'

    def test_getitem_range(self, sample_data):
        """Test __getitem__ for multiple indices"""
        sequences, targets, tickers, timestamps = sample_data
        dataset = FinancialDataset(sequences, targets, tickers, timestamps)
        
        for i in [0, 10, 50, 99]:
            sequence, target, metadata = dataset[i]
            assert metadata['index'] == i

    def test_get_statistics(self, sample_data):
        """Test get_statistics method"""
        sequences, targets, tickers, timestamps = sample_data
        dataset = FinancialDataset(sequences, targets, tickers, timestamps)
        
        stats = dataset.get_statistics()
        
        assert 'n_samples' in stats
        assert 'sequence_length' in stats
        assert 'n_features' in stats
        assert 'target_shape' in stats
        assert 'sequences_mean' in stats
        assert 'sequences_std' in stats
        assert 'targets_mean' in stats
        assert 'targets_std' in stats
        
        assert stats['n_samples'] == 100
        assert stats['sequence_length'] == 60
        assert stats['n_features'] == 10

    def test_transform(self, sample_data):
        """Test dataset with transform"""
        sequences, targets, _, _ = sample_data
        
        # Simple transform: multiply by 2
        def transform(x):
            return x * 2
        
        dataset = FinancialDataset(sequences, targets, transform=transform)
        
        sequence, _, _ = dataset[0]
        
        # Check transform was applied
        expected = torch.FloatTensor(sequences[0]) * 2
        torch.testing.assert_close(sequence, expected)


class TestPhysicsAwareDataset:
    """Test PhysicsAwareDataset class"""

    @pytest.fixture
    def physics_data(self):
        """Create sample data with physics information"""
        np.random.seed(42)
        n_samples = 50
        seq_len = 60
        n_features = 10
        
        sequences = np.random.randn(n_samples, seq_len, n_features)
        targets = np.random.randn(n_samples)
        tickers = [f'TICKER_{i % 3}' for i in range(n_samples)]
        timestamps = [pd.Timestamp('2020-01-01') + pd.Timedelta(days=i) for i in range(n_samples)]
        
        prices = np.abs(np.random.randn(n_samples, seq_len)) * 100 + 100
        returns = np.random.randn(n_samples, seq_len) * 0.02
        volatilities = np.abs(np.random.randn(n_samples, seq_len)) * 0.1 + 0.2
        
        return sequences, targets, tickers, timestamps, prices, returns, volatilities

    def test_physics_dataset_initialization(self, physics_data):
        """Test PhysicsAwareDataset initialization"""
        sequences, targets, tickers, timestamps, prices, returns, volatilities = physics_data
        
        dataset = PhysicsAwareDataset(
            sequences, targets, tickers, timestamps,
            prices, returns, volatilities, dt=1.0
        )
        
        assert len(dataset) == 50
        assert dataset.prices.shape == (50, 60)
        assert dataset.returns.shape == (50, 60)
        assert dataset.volatilities.shape == (50, 60)
        assert dataset.dt == 1.0

    def test_physics_dataset_getitem(self, physics_data):
        """Test PhysicsAwareDataset __getitem__"""
        sequences, targets, tickers, timestamps, prices, returns, volatilities = physics_data
        
        dataset = PhysicsAwareDataset(
            sequences, targets, tickers, timestamps,
            prices, returns, volatilities, dt=1.0
        )
        
        sequence, target, metadata = dataset[0]
        
        # Check physics data in metadata
        assert 'prices' in metadata
        assert 'returns' in metadata
        assert 'volatilities' in metadata
        assert 'dt' in metadata
        
        assert isinstance(metadata['prices'], torch.Tensor)
        assert isinstance(metadata['returns'], torch.Tensor)
        assert isinstance(metadata['volatilities'], torch.Tensor)
        
        assert metadata['prices'].shape == (60,)
        assert metadata['returns'].shape == (60,)
        assert metadata['volatilities'].shape == (60,)
        assert metadata['dt'] == 1.0

    def test_physics_dataset_without_physics_data(self, physics_data):
        """Test PhysicsAwareDataset without optional physics data"""
        sequences, targets, tickers, timestamps, _, _, _ = physics_data
        
        dataset = PhysicsAwareDataset(
            sequences, targets, tickers, timestamps,
            prices=None, returns=None, volatilities=None
        )
        
        sequence, target, metadata = dataset[0]
        
        # Physics data should not be in metadata
        assert 'prices' not in metadata
        assert 'returns' not in metadata
        assert 'volatilities' not in metadata


class TestCollateFunction:
    """Test custom collate function"""

    def test_collate_fn_basic(self):
        """Test collate function with basic dataset"""
        sequences = torch.randn(5, 60, 10)
        targets = torch.randn(5, 1)
        
        batch = [
            (sequences[i], targets[i], {'index': i, 'ticker': f'T{i}', 'timestamp': None})
            for i in range(5)
        ]
        
        batched_seq, batched_targets, batched_metadata = collate_fn_with_metadata(batch)
        
        assert batched_seq.shape == (5, 60, 10)
        assert batched_targets.shape == (5, 1)
        
        assert 'indices' in batched_metadata
        assert 'tickers' in batched_metadata
        assert len(batched_metadata['indices']) == 5
        assert len(batched_metadata['tickers']) == 5

    def test_collate_fn_with_physics(self):
        """Test collate function with physics metadata"""
        sequences = torch.randn(3, 60, 10)
        targets = torch.randn(3, 1)
        prices = torch.randn(3, 60)
        returns = torch.randn(3, 60)
        volatilities = torch.randn(3, 60)
        
        batch = [
            (
                sequences[i],
                targets[i],
                {
                    'index': i,
                    'ticker': f'T{i}',
                    'timestamp': None,
                    'prices': prices[i],
                    'returns': returns[i],
                    'volatilities': volatilities[i],
                    'dt': 1.0
                }
            )
            for i in range(3)
        ]
        
        batched_seq, batched_targets, batched_metadata = collate_fn_with_metadata(batch)
        
        assert 'prices' in batched_metadata
        assert 'returns' in batched_metadata
        assert 'volatilities' in batched_metadata
        assert 'dt' in batched_metadata
        
        assert batched_metadata['prices'].shape == (3, 60)
        assert batched_metadata['returns'].shape == (3, 60)
        assert batched_metadata['volatilities'].shape == (3, 60)
        assert batched_metadata['dt'] == 1.0


class TestDataLoaderCreation:
    """Test dataloader creation"""

    def test_create_dataloaders(self):
        """Test create_dataloaders function"""
        np.random.seed(42)
        
        # Create datasets
        sequences = np.random.randn(100, 60, 10)
        targets = np.random.randn(100)
        
        train_dataset = FinancialDataset(sequences[:70], targets[:70])
        val_dataset = FinancialDataset(sequences[70:85], targets[70:85])
        test_dataset = FinancialDataset(sequences[85:], targets[85:])
        
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=16, num_workers=0, pin_memory=False
        )
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        
        # Check batch sizes
        train_batch = next(iter(train_loader))
        assert len(train_batch) == 3  # sequences, targets, metadata
        assert train_batch[0].shape[0] <= 16  # batch size

    def test_dataloader_iteration(self):
        """Test iterating through dataloader"""
        sequences = np.random.randn(50, 60, 10)
        targets = np.random.randn(50)
        
        dataset = FinancialDataset(sequences, targets)
        loader = DataLoader(dataset, batch_size=10, shuffle=False, collate_fn=collate_fn_with_metadata)
        
        total_samples = 0
        for batch_seq, batch_targets, batch_metadata in loader:
            assert batch_seq.shape[1] == 60  # sequence length
            assert batch_seq.shape[2] == 10  # features
            total_samples += batch_seq.shape[0]
        
        assert total_samples == 50


class TestDatasetEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_dataset(self):
        """Test with empty arrays"""
        sequences = np.array([]).reshape(0, 60, 10)
        targets = np.array([])
        
        dataset = FinancialDataset(sequences, targets)
        
        assert len(dataset) == 0

    def test_single_sample_dataset(self):
        """Test with single sample"""
        sequences = np.random.randn(1, 60, 10)
        targets = np.random.randn(1)
        
        dataset = FinancialDataset(sequences, targets)
        
        assert len(dataset) == 1
        
        sequence, target, metadata = dataset[0]
        assert sequence.shape == (60, 10)
        assert target.shape == (1,)

    def test_mismatched_lengths(self):
        """Test with mismatched sequence and target lengths"""
        sequences = np.random.randn(100, 60, 10)
        targets = np.random.randn(50)  # Mismatched length
        
        # PyTorch will handle this during tensor creation
        # Dataset should still initialize but will have issues on access
        dataset = FinancialDataset(sequences, targets)
        
        # This should work for indices within target range
        sequence, target, metadata = dataset[0]
        assert sequence.shape == (60, 10)

    def test_different_sequence_lengths(self):
        """Test statistics with various sequence lengths"""
        sequences = np.random.randn(50, 120, 5)  # Different seq_len and n_features
        targets = np.random.randn(50, 2)  # Multiple targets
        
        dataset = FinancialDataset(sequences, targets)
        
        stats = dataset.get_statistics()
        assert stats['sequence_length'] == 120
        assert stats['n_features'] == 5
        assert stats['target_shape'] == (2,)

    def test_metadata_partial(self):
        """Test dataset with partial metadata (only tickers, no timestamps)"""
        sequences = np.random.randn(20, 60, 10)
        targets = np.random.randn(20)
        tickers = [f'TICKER_{i}' for i in range(20)]
        
        dataset = FinancialDataset(sequences, targets, tickers=tickers)
        
        sequence, target, metadata = dataset[5]
        assert metadata['ticker'] == 'TICKER_5'
        assert metadata['timestamp'] is None

    def test_torch_tensor_conversion(self):
        """Test that data is properly converted to torch tensors"""
        sequences = np.random.randn(10, 60, 10)
        targets = np.random.randn(10)
        
        dataset = FinancialDataset(sequences, targets)
        
        # Verify internal storage is torch tensors
        assert isinstance(dataset.sequences, torch.Tensor)
        assert isinstance(dataset.targets, torch.Tensor)
        
        # Verify dtype
        assert dataset.sequences.dtype == torch.float32
        assert dataset.targets.dtype == torch.float32


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
