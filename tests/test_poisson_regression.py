"""포아송 회귀분석 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.statistics.effects.regression import (
    build_regression_effect_size_report,
)
from src.statistics.regression.poisson import fit_poisson
from src.statistics.regression.selector import (
    fit_regression_by_level,
)


def make_poisson_dataframe(
    *,
    seed: int = 42,
    size: int = 300,
) -> pd.DataFrame:
    """재현 가능한 포아송 회귀 자료를 생성한다."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=size)
    expected_count = np.exp(0.4 + 0.6 * x)
    y = rng.poisson(expected_count)

    return pd.DataFrame(
        {
            "y": y,
            "x": x,
        }
    )


def test_fit_poisson_returns_incidence_rate_ratios() -> None:
    result = fit_poisson(
        make_poisson_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
        model_id="count_model",
    )

    coefficient_lookup = {coefficient.term: coefficient for coefficient in result.coefficients}

    assert result.model_type == "poisson"
    assert result.model_id == "count_model"
    assert result.sample_size == 300
    assert result.converged is True
    assert coefficient_lookup["x"].exponentiated_estimate is not None
    assert result.fit_statistics["dispersion_ratio"] > 0


def test_count_selector_keeps_poisson_without_overdispersion() -> None:
    result = fit_regression_by_level(
        make_poisson_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
        measurement_level="count",
    )

    assert result.model_type == "poisson"
    assert result.metadata["selected_count_model"] == "poisson"


def test_poisson_rejects_negative_outcomes() -> None:
    dataframe = make_poisson_dataframe()
    dataframe.loc[0, "y"] = -1

    with pytest.raises(
        ValueError,
        match="0 이상의 값",
    ):
        fit_poisson(
            dataframe,
            dependent_variable="y",
            independent_variables=["x"],
        )


def test_poisson_rejects_non_integer_outcomes() -> None:
    dataframe = make_poisson_dataframe()
    dataframe["y"] = dataframe["y"].astype(float)
    dataframe.loc[0, "y"] = 1.5

    with pytest.raises(
        ValueError,
        match="음이 아닌 정수",
    ):
        fit_poisson(
            dataframe,
            dependent_variable="y",
            independent_variables=["x"],
        )


def test_poisson_supports_fixed_effects() -> None:
    dataframe = make_poisson_dataframe()
    dataframe["country"] = np.where(
        np.arange(len(dataframe)) % 2 == 0,
        "KR",
        "US",
    )

    result = fit_poisson(
        dataframe,
        dependent_variable="y",
        independent_variables=["x"],
        fixed_effects=["country"],
    )

    coefficient_terms = {coefficient.term for coefficient in result.coefficients}

    assert result.metadata["fixed_effects"] == ["country"]
    assert result.metadata["fixed_effect_column_count"] == 1
    assert "country_US" in coefficient_terms


def test_poisson_effect_size_uses_incidence_rate_ratio() -> None:
    result = fit_poisson(
        make_poisson_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
    )

    report = build_regression_effect_size_report(result)

    assert report.model_type == "poisson"
    assert any(effect.effect_type == "incidence_rate_ratio" for effect in report.effects)
