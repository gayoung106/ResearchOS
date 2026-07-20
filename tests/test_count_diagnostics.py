"""Poisson 및 Negative Binomial 진단 테스트."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.context import ResearchContext
from src.pipeline.regression_diagnostics_step import (
    RegressionDiagnosticsStep,
)
from src.pipeline.runtime import PipelineRuntime
from src.statistics.diagnostics.count import (
    build_count_diagnostics,
    calculate_count_multicollinearity,
    calculate_count_predictions,
)
from src.statistics.regression.count import (
    fit_count_regression,
)
from src.statistics.regression.poisson import (
    fit_poisson,
)


def make_poisson_result():
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=300)
    x2 = rng.normal(size=300)
    mean = np.exp(0.3 + 0.5 * x1 - 0.3 * x2)
    y = rng.poisson(mean)

    return fit_poisson(
        pd.DataFrame(
            {
                "y": y,
                "x1": x1,
                "x2": x2,
            }
        ),
        dependent_variable="y",
        independent_variables=[
            "x1",
            "x2",
        ],
        model_id="main_model",
    )


def make_negative_binomial_result():
    rng = np.random.default_rng(2026)
    x1 = rng.normal(size=500)
    x2 = rng.normal(size=500)
    mean = np.exp(0.4 + 0.5 * x1 - 0.2 * x2)
    alpha = 1.3
    multiplier = rng.gamma(
        shape=1 / alpha,
        scale=alpha,
        size=len(x1),
    )
    y = rng.poisson(mean * multiplier)

    return fit_count_regression(
        pd.DataFrame(
            {
                "y": y,
                "x1": x1,
                "x2": x2,
            }
        ),
        dependent_variable="y",
        independent_variables=[
            "x1",
            "x2",
        ],
        model_id="main_model",
    )


def test_count_multicollinearity_contains_vif() -> None:
    diagnostics = calculate_count_multicollinearity(make_poisson_result())

    assert len(diagnostics) == 2
    assert all(item.vif is not None for item in diagnostics)


def test_poisson_prediction_metrics() -> None:
    result = make_poisson_result()
    metrics, observations = calculate_count_predictions(result)

    assert metrics.mean_absolute_error >= 0
    assert metrics.root_mean_squared_error >= 0
    assert 0 <= metrics.observed_zero_proportion <= 1
    assert 0 <= metrics.predicted_zero_proportion <= 1
    assert len(observations) == result.sample_size
    assert {
        "actual",
        "predicted",
        "pearson_residual",
        "deviance_residual",
        "leverage",
        "any_diagnostic_flag",
    }.issubset(observations.columns)


def test_build_poisson_diagnostics() -> None:
    report = build_count_diagnostics(make_poisson_result())

    assert report.model_id == "main_model"
    assert report.model_type == "poisson"
    assert report.sample_size == 300
    assert report.summary["root_mean_squared_error"] >= 0
    assert report.summary["pearson_dispersion_ratio"] >= 0
    assert report.summary["residual_degrees_of_freedom"] > 0
    assert "dispersion_ratio" in report.summary


def test_build_negative_binomial_diagnostics() -> None:
    result = make_negative_binomial_result()

    assert result.model_type == "negative_binomial"

    report = build_count_diagnostics(result)

    assert report.model_type == "negative_binomial"
    assert report.sample_size == 500
    assert report.summary["alpha"] > 0
    assert report.summary["pearson_dispersion_ratio"] >= 0
    assert len(report.observations) == 500


def test_pipeline_step_outputs_count_files(
    tmp_path: Path,
) -> None:
    runtime = PipelineRuntime()
    runtime.set_artifact(
        "regression_result:main_model",
        make_negative_binomial_result(),
    )

    step_result = RegressionDiagnosticsStep(
        runtime,
        model_id="main_model",
    ).run(
        ResearchContext(project_name="Count 진단 테스트"),
        tmp_path,
    )

    assert step_result.success is True
    assert len(step_result.output_files) == 4
    assert all(Path(path).exists() for path in step_result.output_files)

    report = runtime.get_artifact("regression_diagnostics:main_model")
    assert report.model_type == "negative_binomial"
    assert report.sample_size == 500
