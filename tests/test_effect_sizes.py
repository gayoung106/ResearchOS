"""회귀 효과크기 엔진 테스트."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.context import ResearchContext
from src.pipeline.effect_size_step import RegressionEffectSizeStep
from src.pipeline.runtime import PipelineRuntime
from src.statistics.effects.regression import (
    build_regression_effect_size_report,
)
from src.statistics.regression.binary_logit import (
    fit_binary_logit,
)
from src.statistics.regression.ols import fit_ols
from src.statistics.regression.ordered_logit import (
    fit_ordered_logit,
)


def test_ols_effect_sizes() -> None:
    rng = np.random.default_rng(10)
    x1 = rng.normal(size=250)
    x2 = rng.normal(size=250)
    y = 1 + 2 * x1 - 0.5 * x2 + rng.normal(size=250)

    result = fit_ols(
        pd.DataFrame(
            {
                "y": y,
                "x1": x1,
                "x2": x2,
            }
        ),
        dependent_variable="y",
        independent_variables=["x1", "x2"],
        model_id="main_model",
    )

    report = build_regression_effect_size_report(result)

    effect_types = {effect.effect_type for effect in report.effects}

    assert "standardized_beta" in effect_types
    assert "partial_r_squared" in effect_types
    assert "partial_cohen_f_squared" in effect_types
    assert report.model_effects["model_cohen_f_squared"] > 0


def test_standardized_beta_direction_matches_coefficient() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=180)
    y = 3 - 1.5 * x + rng.normal(size=180)

    result = fit_ols(
        pd.DataFrame({"y": y, "x": x}),
        dependent_variable="y",
        independent_variables=["x"],
    )
    report = build_regression_effect_size_report(result)

    beta = next(
        effect
        for effect in report.effects
        if effect.term == "x" and effect.effect_type == "standardized_beta"
    )

    assert beta.estimate is not None
    assert beta.estimate < 0


def test_binary_logit_odds_ratio_and_marginal_effect() -> None:
    rng = np.random.default_rng(12)
    x = rng.normal(size=500)
    probability = 1 / (1 + np.exp(-(-0.3 + 1.1 * x)))
    y = rng.binomial(1, probability)

    result = fit_binary_logit(
        pd.DataFrame({"y": y, "x": x}),
        dependent_variable="y",
        independent_variables=["x"],
    )
    report = build_regression_effect_size_report(result)

    effect_types = {effect.effect_type for effect in report.effects}

    assert "odds_ratio" in effect_types
    assert "average_marginal_effect" in effect_types


def test_ordered_logit_returns_odds_ratio() -> None:
    rng = np.random.default_rng(13)
    x = rng.normal(size=350)
    latent = 0.9 * x + rng.logistic(size=350)
    y = pd.cut(
        latent,
        bins=[-np.inf, -0.5, 0.5, np.inf],
        labels=[1, 2, 3],
    ).astype(int)

    result = fit_ordered_logit(
        pd.DataFrame({"y": y, "x": x}),
        dependent_variable="y",
        independent_variables=["x"],
    )
    report = build_regression_effect_size_report(result)

    odds_ratio = next(effect for effect in report.effects if effect.term == "x")

    assert odds_ratio.effect_type == "odds_ratio"
    assert odds_ratio.estimate is not None


def test_effect_size_pipeline_step(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(14)
    x = rng.normal(size=150)
    y = 1 + 2 * x + rng.normal(size=150)
    dataframe = pd.DataFrame({"y": y, "x": x})

    regression_result = fit_ols(
        dataframe,
        dependent_variable="y",
        independent_variables=["x"],
        model_id="main_model",
    )

    runtime = PipelineRuntime(dataframe=dataframe)
    runtime.set_artifact(
        "regression_result:main_model",
        regression_result,
    )

    result = RegressionEffectSizeStep(
        runtime,
        model_id="main_model",
    ).run(
        ResearchContext(project_name="테스트"),
        tmp_path,
    )

    assert result.success is True
    assert len(result.output_files) == 2
    assert all(Path(path).exists() for path in result.output_files)
    assert runtime.get_artifact("effect_size_report:main_model").effects
