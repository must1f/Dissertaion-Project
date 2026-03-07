import numpy as np
import pytest

from src.evaluation.split_manager import SplitManager, SplitConfig


def test_validate_sequence_boundaries_detects_overlap():
    cfg = SplitConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    sm = SplitManager(cfg)
    train, val, test = sm.create_temporal_split(100)

    # sequence_length 50 with horizon 5 would cross into val boundary
    with pytest.raises(ValueError):
        sm.validate_sequence_boundaries(train, val, test, sequence_length=50, forecast_horizon=5)


def test_validate_sequence_boundaries_passes_safe_windows():
    cfg = SplitConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    sm = SplitManager(cfg)
    train, val, test = sm.create_temporal_split(200)

    assert sm.validate_sequence_boundaries(train, val, test, sequence_length=20, forecast_horizon=1)
