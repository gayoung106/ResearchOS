"""Count 회귀 자동선택 및 음이항 회귀 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.count import fit_count_regression
from src.statistics.regression.negative_binomial import fit_negative_binomial


def make_overdispersed_count_dataframe(*, seed: int = 2026, size: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=size)
    mean = np.exp(0.4 + 0.5 * x)
    alpha = 1.2
    mult = rng.gamma(shape=1 / alpha, scale=alpha, size=size)
    y = rng.poisson(mean * mult)
    return pd.DataFrame({"y": y, "x": x})


def test_negative_binomial_returns_incidence_rate_ratios() -> None:
    result = fit_negative_binomial(
        make_overdispersed_count_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
        model_id="nb_model",
    )
    lookup = {c.term: c for c in result.coefficients}
    assert result.model_type == "negative_binomial"
    assert result.converged is True
    assert result.fit_statistics["alpha"] > 0
    assert "alpha" not in lookup
    assert lookup["x"].exponentiated_estimate is not None


def test_count_auto_selects_negative_binomial_for_overdispersion() -> None:
    result = fit_count_regression(
        make_overdispersed_count_dataframe(), dependent_variable="y", independent_variables=["x"]
    )
    assert result.model_type == "negative_binomial"
    assert result.metadata["selected_count_model"] == "negative_binomial"
    assert result.metadata["poisson_dispersion_ratio"] > 1.5
    assert result.metadata["negative_binomial_fitted"] is True


def test_count_auto_selection_metadata_is_recorded() -> None:
    result = fit_count_regression(
        make_overdispersed_count_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
        dispersion_threshold=1.3,
    )
    assert result.metadata["count_model_selection_method"] == "poisson_pearson_dispersion"
    assert result.metadata["dispersion_threshold"] == 1.3
    assert "negative_binomial_aic" in result.metadata


def test_negative_binomial_effect_size_uses_incidence_rate_ratio() -> None:
    result = fit_negative_binomial(
        make_overdispersed_count_dataframe(), dependent_variable="y", independent_variables=["x"]
    )
    report = build_regression_effect_size_report(result)
    assert any(e.effect_type == "incidence_rate_ratio" for e in report.effects)


def test_count_auto_rejects_invalid_dispersion_threshold() -> None:
    with pytest.raises(ValueError, match="1보다 커야"):
        fit_count_regression(
            make_overdispersed_count_dataframe(),
            dependent_variable="y",
            independent_variables=["x"],
            dispersion_threshold=1.0,
        )


def test_negative_binomial_supports_fixed_effects() -> None:
    df = make_overdispersed_count_dataframe()
    df["country"] = np.where(np.arange(len(df)) % 2 == 0, "KR", "US")
    result = fit_negative_binomial(
        df, dependent_variable="y", independent_variables=["x"], fixed_effects=["country"]
    )
    terms = {c.term for c in result.coefficients}
    assert result.metadata["fixed_effect_column_count"] == 1
    assert "country_US" in terms
