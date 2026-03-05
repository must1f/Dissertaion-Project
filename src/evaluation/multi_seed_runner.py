"""
Multi-seed runner for consistent evaluation across seeds.

Wraps the reproducibility MultiSeedRunner to coordinate training/eval functions
and aggregate metrics with bootstrap CIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional
import numpy as np

from src.utils.reproducibility import MultiSeedRunner as CoreMultiSeedRunner
from .statistical_tests import StatisticalTests, PairedTestResult


@dataclass
class SeedResult:
    seed: int
    metrics: Dict[str, float]


@dataclass
class MultiSeedSummary:
    seed_results: List[SeedResult]
    aggregates: Dict[str, Dict[str, float]]  # metric -> {mean, std, ci_lower, ci_upper}


class MultiSeedRunner:
    def __init__(self, base_seed: int = 42, n_seeds: int = 5, ci: float = 0.95):
        self.core = CoreMultiSeedRunner(base_seed=base_seed, n_seeds=n_seeds)
        self.tests = StatisticalTests()
        self.ci = ci

    def run(
        self,
        train_and_evaluate: Callable[[int], Dict[str, float]],
        metric_names: Optional[List[str]] = None,
        baseline_metrics: Optional[Dict[str, List[float]]] = None,
        dm_baseline_errors: Optional[np.ndarray] = None,
        dm_candidate_errors: Optional[np.ndarray] = None,
    ) -> MultiSeedSummary:
        results: List[SeedResult] = []
        for seed in self.core.seeds:
            self.core.set_seed(seed)
            metrics = train_and_evaluate(seed)
            self.core.record_result(seed, metrics)
            results.append(SeedResult(seed=seed, metrics=metrics))

        summary_raw = self.core.get_summary().get("metrics", {})
        aggregates: Dict[str, Dict[str, float]] = {}

        # Add bootstrap CI
        for metric, stats in summary_raw.items():
            values = [r.metrics[metric] for r in results if metric in r.metrics]
            if not values:
                continue
            ci_res = self.tests.bootstrap_confidence_interval(
                np.array(values),
                metric_func=np.mean,
                confidence_level=self.ci,
            )
            aggregates[metric] = {
                "mean": stats["mean"],
                "std": stats["std"],
                "min": stats["min"],
                "max": stats["max"],
                "median": stats["median"],
                "ci_lower": ci_res.ci_lower,
                "ci_upper": ci_res.ci_upper,
            }

            # Optional significance vs provided baseline metric list
            if baseline_metrics and metric in baseline_metrics:
                base_vals = np.array(baseline_metrics[metric])
                try:
                    paired = self.tests.paired_t_test(np.array(values), base_vals)
                    aggregates[metric]["vs_baseline_p"] = paired.p_value
                    aggregates[metric]["vs_baseline_sig"] = paired.significance.value
                    aggregates[metric]["vs_baseline_effect"] = paired.effect_size
                except Exception:
                    pass

        # Optional Diebold-Mariano if error arrays provided (single run)
        if dm_baseline_errors is not None and dm_candidate_errors is not None:
            try:
                dm_res = self.tests.diebold_mariano_test(
                    errors1=dm_baseline_errors,
                    errors2=dm_candidate_errors,
                )
                aggregates["diebold_mariano"] = {
                    "stat": dm_res.test_statistic,
                    "p_value": dm_res.p_value,
                    "better_model": dm_res.better_model,
                    "significance": dm_res.significance.value,
                    "mean_loss_diff": dm_res.mean_loss_diff,
                    "n_observations": dm_res.n_observations,
                }
            except Exception:
                pass

        return MultiSeedSummary(seed_results=results, aggregates=aggregates)
