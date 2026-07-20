"""Random Intercept 혼합효과 진단 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.statistics.diagnostics.mixed_effects import (
    build_mixed_effects_diagnostics,
    calculate_group_residual_summary,
    calculate_mixed_effects_residuals,
    calculate_random_effects,
    mixed_effects_diagnostic_summary_to_dataframe,
    run_mixed_effects_diagnostic_tests,
)
from src.statistics.regression.mixed_effects import fit_random_intercept
from src.statistics.regression.ols import fit_ols


def make_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(20260720)
    group_count = 24
    observations_per_group = 8
    groups = np.repeat(np.arange(group_count), observations_per_group)
    x = rng.normal(size=len(groups))
    random_intercepts = rng.normal(0, 1.2, size=group_count)
    y = 1.5 + 2.0 * x + random_intercepts[groups] + rng.normal(0, 0.5, size=len(groups))
    return pd.DataFrame({"y": y, "x": x, "group": groups})


def make_result():
    return fit_random_intercept(
        make_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="group",
        model_id="mixed_model",
    )


def test_residual_table_has_case_level_values() -> None:
    table = calculate_mixed_effects_residuals(make_result())
    assert len(table) == 192
    assert {"group", "fitted_value", "residual", "standardized_residual"}.issubset(table)
    assert np.isfinite(table["standardized_residual"]).all()


def test_group_residual_summary_has_one_row_per_group() -> None:
    table = calculate_group_residual_summary(make_result())
    assert len(table) == 24
    assert table["group_size"].eq(8).all()
    assert "group_residual_flag" in table


def test_random_effects_are_returned_for_every_group() -> None:
    table = calculate_random_effects(make_result())
    assert len(table) == 24
    assert table["absolute_random_intercept"].is_monotonic_decreasing


def test_diagnostic_tests_include_residual_and_random_effect_normality() -> None:
    tests = run_mixed_effects_diagnostic_tests(make_result())
    assert [item.test_name for item in tests] == [
        "Conditional Residual Jarque-Bera",
        "Random Intercept Jarque-Bera",
    ]
    assert all(item.status in {"PASS", "WARNING", "UNAVAILABLE"} for item in tests)


def test_build_report_contains_core_summary() -> None:
    report = build_mixed_effects_diagnostics(make_result())
    assert report.model_id == "mixed_model"
    assert report.sample_size == 192
    assert report.group_count == 24
    assert report.summary["singular_fit"] is False
    assert report.summary["random_intercept_variance"] > 0


def test_summary_dataframe_uses_item_value_schema() -> None:
    table = mixed_effects_diagnostic_summary_to_dataframe(
        build_mixed_effects_diagnostics(make_result())
    )
    assert list(table.columns) == ["item", "value"]
    assert "intraclass_correlation" in set(table["item"])


def test_small_group_count_marks_random_effect_test_unavailable() -> None:
    data = make_dataframe().iloc[:32].copy()
    data["group"] = np.repeat(np.arange(4), 8)
    result = fit_random_intercept(
        data,
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="group",
    )
    tests = run_mixed_effects_diagnostic_tests(result)
    assert tests[1].status == "UNAVAILABLE"


def test_invalid_model_type_raises() -> None:
    result = fit_ols(
        make_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
    )
    with pytest.raises(ValueError, match="mixed_random_intercept"):
        build_mixed_effects_diagnostics(result)
