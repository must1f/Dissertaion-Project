import numpy as np
import pytest

from src.evaluation.pipeline import EvaluationPipeline, PipelineConfig


def test_pipeline_blocks_oracle_without_flag():
    pipe = EvaluationPipeline(PipelineConfig(generate_plots=False))
    preds = np.array([0.0, 0.01, -0.02])
    rets = np.array([0.0, -0.005, 0.007])

    with pytest.raises(ValueError):
        pipe.evaluate_model(preds, rets, model_name="oracle", is_causal=False, allow_oracle=False)


def test_pipeline_allows_oracle_with_flag():
    pipe = EvaluationPipeline(PipelineConfig(generate_plots=False))
    preds = np.array([0.0, 0.01, -0.02])
    rets = np.array([0.0, -0.005, 0.007])

    result = pipe.evaluate_model(preds, rets, model_name="oracle", is_causal=False, allow_oracle=True)
    assert "sharpe_ratio" in result.metrics
