"""Random Intercept 혼합효과 모형 효과크기 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.statistics.effects.regression import (
    build_regression_effect_size_report,
    effect_size_report_to_dataframe,
    effect_size_summary_to_dataframe,
)
from src.statistics.regression.mixed_effects import fit_random_intercept


def make_mixed_effects_dataframe() -> pd.DataFrame:
    """효과크기 검증용 계층 자료를 생성한다."""
    rng = np.random.default_rng(20260720)
    group_count = 24
    observations_per_group = 8
    groups = np.repeat(np.arange(group_count), observations_per_group)
    x = rng.normal(size=len(groups))
    random_intercepts = rng.normal(0.0, 1.2, size=group_count)
    error = rng.normal(0.0, 0.5, size=len(groups))
    y = 1.5 + 2.0 * x + random_intercepts[groups] + error

    return pd.DataFrame({"y": y, "x": x, "group": groups})


def test_mixed_effects_effect_size_report() -> None:
    result = fit_random_intercept(
        make_mixed_effects_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="group",
        model_id="main_model",
    )

    report = build_regression_effect_size_report(result)

    assert report.model_type == "mixed_random_intercept"
    assert report.metadata["r_squared_method"] == "nakagawa_schielzeth"
    assert report.metadata["group_variable"] == "group"
    assert report.metadata["group_count"] == 24

    standardized_beta = next(
        effect
        for effect in report.effects
        if effect.term == "x" and effect.effect_type == "standardized_beta"
    )

    assert standardized_beta.estimate is not None
    assert standardized_beta.estimate > 0
    assert standardized_beta.p_value is not None


def test_mixed_effects_r_squared_and_icc_are_valid() -> None:
    result = fit_random_intercept(
        make_mixed_effects_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="group",
    )

    report = build_regression_effect_size_report(result)
    marginal = report.model_effects["marginal_r_squared"]
    conditional = report.model_effects["conditional_r_squared"]
    icc = report.model_effects["intraclass_correlation"]

    assert marginal is not None
    assert conditional is not None
    assert icc is not None
    assert 0 <= marginal <= conditional <= 1
    assert 0 <= icc <= 1
    assert report.model_effects["fixed_effect_variance"] > 0
    assert report.model_effects["random_intercept_variance"] > 0
    assert report.model_effects["residual_variance"] > 0


def test_mixed_effects_effect_size_dataframes() -> None:
    result = fit_random_intercept(
        make_mixed_effects_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="group",
    )
    report = build_regression_effect_size_report(result)

    effects = effect_size_report_to_dataframe(report)
    summary = effect_size_summary_to_dataframe(report)

    assert set(effects["effect_type"]) == {"standardized_beta"}
    assert "marginal_r_squared" in set(summary["item"])
    assert "conditional_r_squared" in set(summary["item"])
    assert "intraclass_correlation" in set(summary["item"])


def test_mixed_effects_effect_size_requires_raw_result() -> None:
    result = fit_random_intercept(
        make_mixed_effects_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="group",
    )
    result.raw_result = None

    with pytest.raises(ValueError, match="원본 혼합효과"):
        build_regression_effect_size_report(result)
