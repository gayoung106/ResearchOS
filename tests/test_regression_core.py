"""Regression Core 통합 테스트."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline.context import ResearchContext
from src.pipeline.regression_step import RegressionAnalysisStep
from src.pipeline.runtime import PipelineRuntime
from src.statistics.regression.base import prepare_model_data
from src.statistics.regression.binary_logit import fit_binary_logit
from src.statistics.regression.ols import fit_ols
from src.statistics.regression.ordered_logit import fit_ordered_logit
from src.statistics.regression.selector import fit_regression_by_level


def test_prepare_model_data_drops_missing_cases() -> None:
    dataframe = pd.DataFrame(
        {
            "y": [1, 2, None, 4],
            "x": [1, None, 3, 4],
        }
    )

    prepared = prepare_model_data(
        dataframe,
        "y",
        ["x"],
    )

    assert len(prepared) == 2


def test_ols_recovers_linear_relationship() -> None:
    dataframe = pd.DataFrame(
        {
            "x": np.arange(1, 21, dtype=float),
        }
    )
    dataframe["y"] = 2 + 3 * dataframe["x"]

    result = fit_ols(
        dataframe,
        dependent_variable="y",
        independent_variables=["x"],
        covariance_type="HC3",
    )

    coefficient = next(item for item in result.coefficients if item.term == "x")

    assert result.model_type == "ols"
    assert result.converged is True
    assert coefficient.estimate == pytest.approx(3.0)
    assert result.fit_statistics["r_squared"] == pytest.approx(1.0)


def test_binary_logit_returns_odds_ratio() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=500)
    probability = 1 / (1 + np.exp(-(-0.5 + 1.2 * x)))
    y = rng.binomial(1, probability)
    dataframe = pd.DataFrame({"y": y, "x": x})

    result = fit_binary_logit(
        dataframe,
        dependent_variable="y",
        independent_variables=["x"],
    )

    coefficient = next(item for item in result.coefficients if item.term == "x")

    assert result.model_type == "binary_logit"
    assert result.converged is True
    assert coefficient.exponentiated_estimate is not None
    assert coefficient.exponentiated_estimate > 1


def test_binary_logit_requires_zero_one_outcome() -> None:
    dataframe = pd.DataFrame(
        {
            "y": [1, 2, 1, 2],
            "x": [0.1, 0.2, 0.3, 0.4],
        }
    )

    with pytest.raises(ValueError, match="0과 1"):
        fit_binary_logit(
            dataframe,
            dependent_variable="y",
            independent_variables=["x"],
        )


def test_ordered_logit_runs() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=400)
    latent = 0.8 * x + rng.logistic(size=400)
    y = pd.cut(
        latent,
        bins=[-np.inf, -0.5, 0.5, np.inf],
        labels=[1, 2, 3],
    ).astype(int)
    dataframe = pd.DataFrame({"y": y, "x": x})

    result = fit_ordered_logit(
        dataframe,
        dependent_variable="y",
        independent_variables=["x"],
    )

    assert result.model_type == "ordered_logit"
    assert result.sample_size == 400
    assert any(coefficient.term == "x" for coefficient in result.coefficients)


def test_selector_uses_measurement_level() -> None:
    dataframe = pd.DataFrame(
        {
            "x": np.arange(1, 11, dtype=float),
            "y": np.arange(1, 11, dtype=float) * 2,
        }
    )

    result = fit_regression_by_level(
        dataframe,
        dependent_variable="y",
        independent_variables=["x"],
        measurement_level="continuous",
    )

    assert result.model_type == "ols"


def test_regression_pipeline_step(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "x": np.arange(1, 21, dtype=float),
        }
    )
    dataframe["y"] = 1 + 2 * dataframe["x"]

    runtime = PipelineRuntime(dataframe=dataframe)
    step = RegressionAnalysisStep(
        runtime,
        dependent_variable="y",
        independent_variables=["x"],
        measurement_level="continuous",
        model_id="main_model",
    )

    result = step.run(
        ResearchContext(project_name="테스트"),
        tmp_path,
    )

    assert result.success is True
    assert len(result.output_files) == 2
    assert all(Path(path).exists() for path in result.output_files)
    stored = runtime.get_artifact("regression_result:main_model")
    assert stored.model_type == "ols"
