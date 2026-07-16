"""영과잉 계수형 회귀와 자동선택 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.statistics.effects.regression import (
    build_regression_effect_size_report,
)
from src.statistics.regression.count import fit_count_regression
from src.statistics.regression.zero_inflated_negative_binomial import (
    fit_zero_inflated_negative_binomial,
)
from src.statistics.regression.zero_inflated_poisson import (
    fit_zero_inflated_poisson,
)


def make_zip_dataframe(
    *,
    seed: int = 41,
    size: int = 600,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=size)
    mean = np.exp(0.3 + 0.5 * x)
    y = rng.poisson(mean)
    structural_zero = rng.random(size) < 0.45
    y[structural_zero] = 0

    return pd.DataFrame(
        {
            "y": y,
            "x": x,
        }
    )


def make_zinb_dataframe(
    *,
    seed: int = 77,
    size: int = 800,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=size)
    mean = np.exp(0.4 + 0.5 * x)
    alpha = 1.2
    multiplier = rng.gamma(
        shape=1 / alpha,
        scale=alpha,
        size=size,
    )
    y = rng.poisson(mean * multiplier)
    structural_zero = rng.random(size) < 0.40
    y[structural_zero] = 0

    return pd.DataFrame(
        {
            "y": y,
            "x": x,
        }
    )


def test_fit_zero_inflated_poisson() -> None:
    result = fit_zero_inflated_poisson(
        make_zip_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
    )

    terms = {coefficient.term for coefficient in result.coefficients}

    assert result.model_type == "zero_inflated_poisson"
    assert result.converged is True
    assert "inflate_const" in terms
    assert "x" in terms
    assert result.fit_statistics["predicted_zero_proportion"] > 0


def test_fit_zero_inflated_negative_binomial() -> None:
    result = fit_zero_inflated_negative_binomial(
        make_zinb_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
    )

    terms = {coefficient.term for coefficient in result.coefficients}

    assert result.model_type == "zero_inflated_negative_binomial"
    assert result.converged is True
    assert result.fit_statistics["alpha"] > 0
    assert "alpha" not in terms
    assert "inflate_const" in terms


def test_count_auto_selects_zero_inflated_model() -> None:
    result = fit_count_regression(
        make_zip_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
    )

    assert result.model_type in {
        "zero_inflated_poisson",
        "zero_inflated_negative_binomial",
    }
    assert result.metadata["count_model_selection_method"] == "dispersion_then_zero_inflation_aic"
    assert result.metadata["zero_inflated_candidates_fitted"] is True
    assert result.metadata["aic_improvement_over_baseline"] >= 2
    assert result.metadata["selected_count_model"] == result.model_type


def test_count_auto_selects_zinb_for_overdispersed_zero_inflation() -> None:
    result = fit_count_regression(
        make_zinb_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
    )

    assert result.model_type == "zero_inflated_negative_binomial"
    assert result.metadata["baseline_count_model"] == "negative_binomial"
    assert result.metadata["selected_count_model"] == result.model_type


def test_zero_inflated_effect_size_excludes_inflation_intercept() -> None:
    result = fit_zero_inflated_poisson(
        make_zip_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
    )

    report = build_regression_effect_size_report(result)

    assert report.model_type == "zero_inflated_poisson"
    assert all(not effect.term.startswith("inflate_") for effect in report.effects)
    assert any(effect.effect_type == "incidence_rate_ratio" for effect in report.effects)
