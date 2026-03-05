"""
Tests for Curriculum Scheduler

Validates:
- Schedule computation (linear, cosine, exponential, etc.)
- Phase transitions (warmup → ramp → full)
- Per-constraint weight scaling
- Dynamic/adaptive scheduling
"""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.curriculum_scheduler import (
    CurriculumScheduler,
    CurriculumConfig,
    ConstraintScheduler,
    DynamicCurriculumScheduler,
    ScheduleType,
    create_curriculum_scheduler
)
from src.training.curriculum import (
    CurriculumScheduler as SimpleCurriculumScheduler,
    AdaptiveCurriculumScheduler
)


class TestCurriculumScheduler:
    """Tests for the main CurriculumScheduler class."""

    def test_initialization(self):
        """Test scheduler initializes with correct parameters."""
        scheduler = CurriculumScheduler(
            warmup_epochs=10,
            ramp_epochs=20,
            schedule='linear',
            final_physics_weight=1.0
        )

        assert scheduler.warmup_epochs == 10
        assert scheduler.ramp_epochs == 20
        assert scheduler.final_physics_weight == 1.0
        assert scheduler.current_epoch == 0

    def test_warmup_phase(self):
        """During warmup, physics weight should be minimal."""
        scheduler = CurriculumScheduler(
            warmup_epochs=10,
            ramp_epochs=20,
            min_physics_weight=0.0
        )

        # During warmup (epochs 0-9), weight should be 0
        for epoch in range(10):
            weight = scheduler.get_physics_weight(epoch)
            assert weight == 0.0, f"Epoch {epoch}: expected 0.0, got {weight}"
            assert scheduler.get_phase(epoch) == 'warmup'

    def test_ramp_phase(self):
        """During ramp, physics weight should increase."""
        scheduler = CurriculumScheduler(
            warmup_epochs=10,
            ramp_epochs=20,
            schedule='linear',
            final_physics_weight=1.0,
            min_physics_weight=0.0
        )

        weights = []
        for epoch in range(10, 30):
            weight = scheduler.get_physics_weight(epoch)
            weights.append(weight)
            assert scheduler.get_phase(epoch) == 'ramp'

        # Weights should be monotonically increasing
        for i in range(len(weights) - 1):
            assert weights[i+1] >= weights[i], "Weights should increase during ramp"

        # Last ramp epoch should be near final weight
        assert weights[-1] >= 0.9, f"End of ramp should be near 1.0, got {weights[-1]}"

    def test_full_phase(self):
        """After ramp, physics weight should be at final value."""
        scheduler = CurriculumScheduler(
            warmup_epochs=10,
            ramp_epochs=20,
            final_physics_weight=1.0
        )

        # After ramp (epoch 30+), weight should be at final value
        for epoch in range(30, 50):
            weight = scheduler.get_physics_weight(epoch)
            assert weight == 1.0, f"Epoch {epoch}: expected 1.0, got {weight}"
            assert scheduler.get_phase(epoch) == 'full'

    def test_linear_schedule(self):
        """Linear schedule should increase at constant rate."""
        scheduler = CurriculumScheduler(
            warmup_epochs=0,
            ramp_epochs=10,
            schedule='linear',
            final_physics_weight=1.0,
            min_physics_weight=0.0
        )

        weights = [scheduler.get_physics_weight(e) for e in range(10)]

        # Check linear increase
        for i in range(1, len(weights)):
            diff = weights[i] - weights[i-1]
            expected_diff = 1.0 / 10
            assert abs(diff - expected_diff) < 0.01, f"Linear diff should be ~0.1, got {diff}"

    def test_cosine_schedule(self):
        """Cosine schedule should have smooth S-curve."""
        scheduler = CurriculumScheduler(
            warmup_epochs=0,
            ramp_epochs=20,
            schedule='cosine',
            final_physics_weight=1.0,
            min_physics_weight=0.0
        )

        weights = [scheduler.get_physics_weight(e) for e in range(20)]

        # Start should increase slowly
        assert weights[1] < 0.05, "Cosine should start slowly"
        # End should increase slowly too
        assert weights[-1] - weights[-2] < 0.1, "Cosine should end slowly"
        # Middle should increase fastest
        mid_diff = weights[10] - weights[9]
        start_diff = weights[1] - weights[0]
        assert mid_diff > start_diff, "Middle should increase faster than start"

    def test_step_schedule(self):
        """Step schedule should have discrete jumps."""
        scheduler = CurriculumScheduler(
            warmup_epochs=0,
            ramp_epochs=16,
            schedule='step',
            final_physics_weight=1.0,
            min_physics_weight=0.0
        )

        weights = [scheduler.get_physics_weight(e) for e in range(16)]

        # Should have distinct plateaus
        unique_weights = len(set([round(w, 2) for w in weights]))
        assert unique_weights <= 5, f"Step should have few unique values, got {unique_weights}"

    def test_exponential_schedule(self):
        """Exponential schedule should start slow, end fast."""
        scheduler = CurriculumScheduler(
            warmup_epochs=0,
            ramp_epochs=20,
            schedule='exponential',
            final_physics_weight=1.0,
            min_physics_weight=0.0
        )

        weights = [scheduler.get_physics_weight(e) for e in range(20)]

        # First half should be less than 0.5 total progress
        first_half_max = max(weights[:10])
        assert first_half_max < 0.5, "Exponential first half should be < 0.5"

    def test_step_method(self):
        """step() should advance current_epoch and record history."""
        scheduler = CurriculumScheduler(warmup_epochs=5, ramp_epochs=10)

        assert scheduler.current_epoch == 0

        scheduler.step()
        assert scheduler.current_epoch == 1
        assert len(scheduler.history) == 1

        for _ in range(9):
            scheduler.step()
        assert scheduler.current_epoch == 10
        assert len(scheduler.history) == 10

    def test_get_state(self):
        """get_state() should return current curriculum state."""
        scheduler = CurriculumScheduler(
            warmup_epochs=5,
            ramp_epochs=10,
            schedule='linear'
        )

        state = scheduler.get_state()

        assert state.epoch == 0
        assert state.phase == 'warmup'
        assert state.physics_weight == 0.0
        assert state.schedule_type == 'linear'

    def test_from_config(self):
        """Should create scheduler from config object."""
        config = CurriculumConfig(
            warmup_epochs=15,
            ramp_epochs=25,
            schedule='cosine',
            final_physics_weight=0.8
        )

        scheduler = CurriculumScheduler.from_config(config)

        assert scheduler.warmup_epochs == 15
        assert scheduler.ramp_epochs == 25
        assert scheduler.final_physics_weight == 0.8

    def test_reset(self):
        """reset() should restore initial state."""
        scheduler = CurriculumScheduler(warmup_epochs=5, ramp_epochs=10)

        for _ in range(10):
            scheduler.step()

        assert scheduler.current_epoch == 10
        assert len(scheduler.history) == 10

        scheduler.reset()

        assert scheduler.current_epoch == 0
        assert len(scheduler.history) == 0


class TestConstraintScheduler:
    """Tests for per-constraint staggered scheduling."""

    def test_staggered_creation(self):
        """Test creating staggered constraint scheduler."""
        scheduler = ConstraintScheduler.create_staggered(
            constraints=['gbm', 'ou', 'bs'],
            base_warmup=5,
            stagger=5,
            ramp_epochs=10
        )

        assert 'gbm' in scheduler.schedules
        assert 'ou' in scheduler.schedules
        assert 'bs' in scheduler.schedules

    def test_staggered_timing(self):
        """Earlier constraints should ramp up before later ones."""
        scheduler = ConstraintScheduler.create_staggered(
            constraints=['gbm', 'ou', 'bs'],
            base_warmup=5,
            stagger=5,
            ramp_epochs=10
        )

        # At epoch 6 (just after gbm warmup)
        weights = scheduler.get_weights(epoch=6)
        assert weights['gbm'] > 0, "GBM should have started ramping"
        assert weights['ou'] == 0, "OU should still be in warmup"
        assert weights['bs'] == 0, "BS should still be in warmup"

        # At epoch 11 (after ou warmup starts)
        weights = scheduler.get_weights(epoch=11)
        assert weights['gbm'] > weights['ou'], "GBM should be ahead of OU"
        assert weights['ou'] > 0, "OU should have started"
        assert weights['bs'] == 0, "BS still in warmup"

    def test_step_advances_all(self):
        """step() should advance all constraint schedulers."""
        scheduler = ConstraintScheduler.create_staggered(
            constraints=['gbm', 'ou'],
            base_warmup=2,
            stagger=2,
            ramp_epochs=5
        )

        initial_epoch = scheduler.current_epoch
        scheduler.step()

        assert scheduler.current_epoch == initial_epoch + 1


class TestDynamicCurriculumScheduler:
    """Tests for dynamic/adaptive curriculum scheduling."""

    def test_update_from_metrics_slows_on_instability(self):
        """Scheduler should slow down when training is unstable."""
        scheduler = DynamicCurriculumScheduler(
            warmup_epochs=5,
            ramp_epochs=10,
            stability_threshold=0.5
        )

        initial_factor = scheduler.adjustment_factor

        # Simulate unstable training
        scheduler.update_from_metrics(
            loss=0.5,
            stability_score=0.3,  # Below threshold
            prev_loss=0.4
        )

        assert scheduler.adjustment_factor < initial_factor, \
            "Should slow down on instability"

    def test_update_from_metrics_speeds_on_good_progress(self):
        """Scheduler should speed up when loss improving well."""
        scheduler = DynamicCurriculumScheduler(
            warmup_epochs=5,
            ramp_epochs=10,
            stability_threshold=0.5
        )

        initial_factor = scheduler.adjustment_factor

        # Simulate good progress
        scheduler.update_from_metrics(
            loss=0.1,
            stability_score=0.8,  # Above threshold
            prev_loss=0.2  # Good improvement
        )

        assert scheduler.adjustment_factor >= initial_factor, \
            "Should not slow down on good progress"


class TestSimpleCurriculumScheduler:
    """Tests for the simpler curriculum scheduler in curriculum.py."""

    def test_basic_scheduling(self):
        """Test basic weight scheduling."""
        scheduler = SimpleCurriculumScheduler(
            initial_lambda_gbm=0.0,
            final_lambda_gbm=0.1,
            initial_lambda_ou=0.0,
            final_lambda_ou=0.1,
            warmup_epochs=5,
            total_epochs=50,
            strategy='linear'
        )

        # During warmup
        weights = scheduler.step(epoch=2)
        assert weights['lambda_gbm'] == 0.0
        assert weights['lambda_ou'] == 0.0

        # After warmup
        weights = scheduler.step(epoch=10)
        assert weights['lambda_gbm'] > 0.0
        assert weights['lambda_ou'] > 0.0

    def test_all_strategies(self):
        """All scheduling strategies should work."""
        strategies = ['linear', 'exponential', 'cosine', 'step']

        for strategy in strategies:
            scheduler = SimpleCurriculumScheduler(
                initial_lambda_gbm=0.0,
                final_lambda_gbm=0.1,
                warmup_epochs=5,
                total_epochs=50,
                strategy=strategy
            )

            # Should not raise
            weights = scheduler.step(epoch=25)
            assert 0.0 <= weights['lambda_gbm'] <= 0.1, \
                f"Strategy {strategy} produced invalid weight: {weights['lambda_gbm']}"


class TestAdaptiveCurriculumScheduler:
    """Tests for adaptive curriculum based on validation loss."""

    def test_adaptive_adjustment(self):
        """Adaptive scheduler should adjust based on validation loss."""
        scheduler = AdaptiveCurriculumScheduler(
            initial_lambda_gbm=0.0,
            final_lambda_gbm=0.1,
            warmup_epochs=5,
            total_epochs=50,
            patience=3,
            threshold=0.01
        )

        # Good improvement
        weights1 = scheduler.step_adaptive(epoch=10, val_loss=0.5)
        weights2 = scheduler.step_adaptive(epoch=11, val_loss=0.4)  # Better

        assert scheduler.adjustment_factor >= 1.0, "Should not penalize on improvement"

    def test_patience_mechanism(self):
        """Should reduce speed after patience epochs without improvement."""
        scheduler = AdaptiveCurriculumScheduler(
            initial_lambda_gbm=0.0,
            final_lambda_gbm=0.1,
            warmup_epochs=2,
            total_epochs=20,
            patience=2,
            threshold=0.001
        )

        initial_factor = scheduler.adjustment_factor

        # No improvement for patience epochs
        scheduler.step_adaptive(epoch=5, val_loss=0.5)
        scheduler.step_adaptive(epoch=6, val_loss=0.5)  # Same
        scheduler.step_adaptive(epoch=7, val_loss=0.5)  # Still same

        # After patience, factor should decrease
        assert scheduler.adjustment_factor < initial_factor, \
            "Should slow down after patience exceeded"


class TestFactoryFunction:
    """Tests for create_curriculum_scheduler factory."""

    def test_create_from_dict(self):
        """Should create scheduler from config dict."""
        config = {
            'warmup_epochs': 8,
            'ramp_epochs': 15,
            'schedule': 'cosine',
            'final_physics_weight': 0.9
        }

        scheduler = create_curriculum_scheduler(config, total_epochs=100)

        assert scheduler.warmup_epochs == 8
        assert scheduler.ramp_epochs == 15
        assert scheduler.final_physics_weight == 0.9

    def test_defaults(self):
        """Should use defaults for missing config values."""
        scheduler = create_curriculum_scheduler({}, total_epochs=50)

        assert scheduler.warmup_epochs == 10  # Default
        assert scheduler.ramp_epochs == 20  # Default


def test_curriculum_imports():
    """Verify all curriculum exports are importable."""
    from src.training import (
        CurriculumScheduler,
        CurriculumConfig,
        CurriculumState,
        ConstraintScheduler,
        DynamicCurriculumScheduler,
        ScheduleType,
        create_curriculum_scheduler
    )

    assert callable(CurriculumScheduler)
    assert callable(create_curriculum_scheduler)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
