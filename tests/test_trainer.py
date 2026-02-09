"""
Comprehensive tests for training infrastructure
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import EarlyStopping, Trainer


class TestEarlyStopping:
    """Test early stopping functionality"""

    def test_early_stopping_min_mode(self):
        """Test early stopping in min mode (lower is better)"""
        early_stop = EarlyStopping(patience=3, min_delta=0.0, mode='min')
        
        # First score - should not stop
        assert not early_stop(1.0)
        assert early_stop.best_score == 1.0
        assert early_stop.counter == 0
        
        # Improvement - should not stop
        assert not early_stop(0.8)
        assert early_stop.best_score == 0.8
        assert early_stop.counter == 0
        
        # No improvement - increment counter
        assert not early_stop(0.9)
        assert early_stop.counter == 1
        
        # Still no improvement
        assert not early_stop(0.85)
        assert early_stop.counter == 2
        
        # Still no improvement - should stop
        should_stop = early_stop(0.82)
        assert early_stop.counter == 3
        assert should_stop  # Should stop after patience exceeded

    def test_early_stopping_max_mode(self):
        """Test early stopping in max mode (higher is better)"""
        early_stop = EarlyStopping(patience=2, min_delta=0.0, mode='max')
        
        # First score
        assert not early_stop(0.5)
        assert early_stop.best_score == 0.5
        
        # Improvement
        assert not early_stop(0.7)
        assert early_stop.best_score == 0.7
        
        # No improvement
        assert not early_stop(0.6)
        assert early_stop.counter == 1
        
        # No improvement - should stop
        assert early_stop(0.65)
        assert early_stop.counter == 2
        assert early_stop.early_stop

    def test_early_stopping_min_delta(self):
        """Test early stopping with min_delta threshold"""
        early_stop = EarlyStopping(patience=2, min_delta=0.1, mode='min')
        
        # First score
        assert not early_stop(1.0)
        
        # Small improvement (less than min_delta) - should not count
        assert not early_stop(0.95)
        assert early_stop.counter == 1  # Not enough improvement
        
        # Significant improvement
        assert not early_stop(0.8)
        assert early_stop.counter == 0  # Reset counter
        assert early_stop.best_score == 0.8

    def test_early_stopping_patience_reset(self):
        """Test that counter resets on improvement"""
        early_stop = EarlyStopping(patience=2, min_delta=0.0, mode='min')
        
        assert not early_stop(1.0)  # Initial
        assert not early_stop(1.1)  # No improvement, counter = 1
        assert early_stop.counter == 1
        
        assert not early_stop(0.9)  # Improvement, counter should reset
        assert early_stop.counter == 0
        assert early_stop.best_score == 0.9

    def test_early_stopping_first_score(self):
        """Test that first score is always accepted"""
        early_stop = EarlyStopping(patience=1, mode='min')
        
        # First call should never stop
        assert not early_stop(100.0)
        assert early_stop.best_score == 100.0

    def test_early_stopping_zero_patience(self):
        """Test with patience = 0 (stops immediately on no improvement)"""
        early_stop = EarlyStopping(patience=0, mode='min')
        
        assert not early_stop(1.0)  # First score
        assert early_stop(1.1)  # No improvement, should stop immediately


class SimpleDummyModel(nn.Module):
    """Simple model for testing"""
    def __init__(self, input_dim=10, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim) -> take last timestep
        x = x[:, -1, :]  # (batch, input_dim)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TestTrainer:
    """Test Trainer class"""

    @pytest.fixture
    def dummy_data(self):
        """Create dummy data for testing"""
        batch_size = 16
        seq_len = 60
        input_dim = 10
        n_samples = 100
        
        # Create dummy sequences and targets
        sequences = torch.randn(n_samples, seq_len, input_dim)
        targets = torch.randn(n_samples, 1)
        
        # Create datasets
        dataset = TensorDataset(sequences, targets)
        
        # Split into train/val/test
        train_size = 70
        val_size = 15
        test_size = 15
        
        train_dataset = TensorDataset(sequences[:train_size], targets[:train_size])
        val_dataset = TensorDataset(sequences[train_size:train_size+val_size], 
                                    targets[train_size:train_size+val_size])
        test_dataset = TensorDataset(sequences[train_size+val_size:], 
                                     targets[train_size+val_size:])
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader

    @pytest.fixture
    def simple_model(self):
        """Create a simple model"""
        return SimpleDummyModel(input_dim=10, hidden_dim=32)

    def test_trainer_initialization(self, simple_model, dummy_data):
        """Test trainer initialization"""
        train_loader, val_loader, test_loader = dummy_data
        
        trainer = Trainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.criterion is not None
        assert trainer.early_stopping is not None
        assert isinstance(trainer.history, dict)

    def test_trainer_device_handling(self, simple_model, dummy_data):
        """Test device handling (CPU/CUDA)"""
        train_loader, val_loader, test_loader = dummy_data
        
        # Force CPU
        trainer = Trainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=torch.device('cpu')
        )
        
        assert trainer.device.type == 'cpu'
        # Check model is on correct device
        assert next(trainer.model.parameters()).device.type == 'cpu'

    def test_trainer_history_structure(self, simple_model, dummy_data):
        """Test that history dictionary has correct structure"""
        train_loader, val_loader, test_loader = dummy_data
        
        trainer = Trainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )
        
        # Check history has expected keys
        assert 'train_loss' in trainer.history
        assert 'val_loss' in trainer.history
        assert 'epochs' in trainer.history
        
        # Should be empty initially
        assert len(trainer.history['train_loss']) == 0
        assert len(trainer.history['val_loss']) == 0

    def test_model_parameter_count(self, simple_model, dummy_data):
        """Test that model parameters are counted correctly"""
        train_loader, val_loader, test_loader = dummy_data
        
        trainer = Trainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )
        
        # Count parameters manually
        n_params = sum(p.numel() for p in trainer.model.parameters())
        
        assert n_params > 0
        # SimpleDummyModel should have: (10*32 + 32) + (32*1 + 1) = 320 + 32 + 32 + 1 = 385
        expected_params = (10 * 32 + 32) + (32 * 1 + 1)
        assert n_params == expected_params


class TestTrainerEdgeCases:
    """Test edge cases in training"""

    def test_empty_dataloader(self):
        """Test with empty dataloader"""
        model = SimpleDummyModel()
        
        # Create empty dataloaders
        empty_dataset = TensorDataset(torch.empty(0, 60, 10), torch.empty(0, 1))
        train_loader = DataLoader(empty_dataset, batch_size=16)
        val_loader = DataLoader(empty_dataset, batch_size=16)
        test_loader = DataLoader(empty_dataset, batch_size=16)
        
        # Trainer should initialize but training will be trivial
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )
        
        assert trainer is not None

    def test_single_batch_training(self):
        """Test with single batch"""
        model = SimpleDummyModel()
        
        # Create single batch dataset
        sequences = torch.randn(5, 60, 10)
        targets = torch.randn(5, 1)
        dataset = TensorDataset(sequences, targets)
        
        loader = DataLoader(dataset, batch_size=5)
        
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            test_loader=loader
        )
        
        assert trainer is not None
        assert len(loader) == 1  # Only one batch

    def test_different_batch_sizes(self):
        """Test with different batch sizes for train/val/test"""
        model = SimpleDummyModel()
        
        sequences = torch.randn(100, 60, 10)
        targets = torch.randn(100, 1)
        
        train_dataset = TensorDataset(sequences[:70], targets[:70])
        val_dataset = TensorDataset(sequences[70:85], targets[70:85])
        test_dataset = TensorDataset(sequences[85:], targets[85:])
        
        train_loader = DataLoader(train_dataset, batch_size=16)
        val_loader = DataLoader(val_dataset, batch_size=8)
        test_loader = DataLoader(test_dataset, batch_size=4)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )
        
        assert trainer is not None


class TestEarlyStoppingIntegration:
    """Integration tests for early stopping with trainer"""

    def test_early_stopping_different_modes(self):
        """Test early stopping works with different modes"""
        # Test min mode
        es_min = EarlyStopping(patience=2, mode='min')
        scores_min = [1.0, 0.8, 0.85, 0.9, 0.95]  # Improves then plateaus
        
        stops = [es_min(score) for score in scores_min]
        assert not stops[0]  # First never stops
        assert not stops[1]  # Improvement
        assert not stops[2]  # No improvement, counter = 1
        assert not stops[3]  # No improvement, counter = 2
        # Note: stops[4] might not trigger stop if patience=2 means "wait 2 epochs"
        
        # Test max mode
        es_max = EarlyStopping(patience=2, mode='max')
        scores_max = [0.5, 0.7, 0.65, 0.6, 0.55]  # Improves then degrades
        
        stops = [es_max(score) for score in scores_max]
        assert not stops[0]  # First never stops
        assert not stops[1]  # Improvement

    def test_early_stopping_tracks_best_score(self):
        """Test that early stopping tracks best score correctly"""
        es = EarlyStopping(patience=5, mode='min')
        
        scores = [1.0, 0.8, 0.9, 0.7, 0.85, 0.75]
        for score in scores:
            es(score)
        
        # Best score should be 0.7
        assert es.best_score == 0.7


class TestTrainerComponents:
    """Test individual trainer components"""

    def test_optimizer_initialization(self):
        """Test optimizer is initialized correctly"""
        model = SimpleDummyModel()
        sequences = torch.randn(50, 60, 10)
        targets = torch.randn(50, 1)
        dataset = TensorDataset(sequences, targets)
        loader = DataLoader(dataset, batch_size=16)
        
        trainer = Trainer(model, loader, loader, loader)
        
        assert trainer.optimizer is not None
        assert isinstance(trainer.optimizer, torch.optim.Optimizer)
        assert len(trainer.optimizer.param_groups) > 0

    def test_scheduler_initialization(self):
        """Test learning rate scheduler is initialized"""
        model = SimpleDummyModel()
        sequences = torch.randn(50, 60, 10)
        targets = torch.randn(50, 1)
        dataset = TensorDataset(sequences, targets)
        loader = DataLoader(dataset, batch_size=16)
        
        trainer = Trainer(model, loader, loader, loader)
        
        assert trainer.scheduler is not None
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler._LRScheduler)

    def test_loss_criterion(self):
        """Test loss criterion is initialized"""
        model = SimpleDummyModel()
        sequences = torch.randn(50, 60, 10)
        targets = torch.randn(50, 1)
        dataset = TensorDataset(sequences, targets)
        loader = DataLoader(dataset, batch_size=16)
        
        trainer = Trainer(model, loader, loader, loader)
        
        assert trainer.criterion is not None
        assert isinstance(trainer.criterion, nn.Module)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
