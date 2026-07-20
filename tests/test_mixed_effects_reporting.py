"""Random Intercept 혼합효과모형 논문용 보고서 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.reporting.regression import build_regression_publication_report
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.mixed_effects import fit_random_intercept


def make_mixed_effects_dataframe() -> pd.DataFrame:
    """보고서 검증용 계층 자료를 생성한다."""
    rng = np.random.default_rng(20260720)
    group_count = 24
    observations_per_group = 8
    groups = np.repeat(np.arange(group_count), observations_per_group)
    x = rng.normal(size=len(groups))
    random_intercepts = rng.normal(0.0, 1.2, size=group_count)
    error = rng.normal(0.0, 0.5, size=len(groups))
    y = 1.5 + 2.0 * x + random_intercepts[groups] + error
    return pd.DataFrame({"y": y, "x": x, "group": groups})


def test_mixed_effects_publication_report_contains_model_information() -> None:
    result = fit_random_intercept(
        make_mixed_effects_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="group",
        model_id="main_model",
        reml=False,
    )
    effect_report = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effect_report)

    assert report.model_type == "mixed_random_intercept"
    assert report.metadata["group_variable"] == "group"
    assert report.metadata["group_count"] == 24
    assert report.metadata["converged"] is True
    assert report.metadata["optimizer"] == "lbfgs"
    assert report.metadata["reml"] is False

    summary_items = set(report.model_summary["항목"])
    assert "random_intercept_variance" in summary_items
    assert "residual_variance" in summary_items
    assert "intraclass_correlation" in summary_items
    assert "marginal_r_squared" in summary_items
    assert "conditional_r_squared" in summary_items


def test_mixed_effects_publication_table_includes_standardized_beta() -> None:
    result = fit_random_intercept(
        make_mixed_effects_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="group",
    )
    effect_report = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effect_report)
    x_row = report.publication_table.loc[report.publication_table["변수"] == "x"].iloc[0]

    assert x_row["구분"] == "predictor"
    assert x_row["표준화 β"] > 0
    assert x_row["유의성"] == "***"


def test_mixed_effects_narrative_and_notes_are_generated() -> None:
    result = fit_random_intercept(
        make_mixed_effects_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="group",
    )
    effect_report = build_regression_effect_size_report(result)
    report = build_regression_publication_report(result, effect_report)

    assert "Random Intercept 혼합효과모형" in report.narrative
    assert "24개 집단" in report.narrative
    assert "ICC=" in report.narrative
    assert "marginal R²=" in report.narrative
    assert "conditional R²=" in report.narrative
    assert "Random Intercept 분산" in report.narrative
    assert "모형은 수렴하였다" in report.narrative
    assert any("혼합효과모형의 고정효과" in note for note in report.notes)
    assert any("ICC" in note for note in report.notes)
