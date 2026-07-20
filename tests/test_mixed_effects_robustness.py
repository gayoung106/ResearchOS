"""Random Intercept optimizer 강건성 분석 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.statistics.robustness.mixed_effects import (
    build_mixed_effects_robustness_report,
    coefficient_comparison_to_dataframe,
    model_comparison_to_dataframe,
    robustness_summary_to_dataframe,
    stability_summary_to_dataframe,
)


def make_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(20260720)
    groups = np.repeat(np.arange(18), 8)
    x = rng.normal(size=len(groups))
    random_intercepts = rng.normal(scale=0.8, size=18)
    y = 1.2 + 1.8 * x + random_intercepts[groups] + rng.normal(scale=0.45, size=len(groups))
    return pd.DataFrame({"y": y, "x": x, "group": groups})


def test_build_mixed_effects_robustness_report() -> None:
    report = build_mixed_effects_robustness_report(
        make_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="group",
        optimizers=("lbfgs", "bfgs"),
    )

    assert report.group_variable == "group"
    assert report.summary["requested_optimizer_count"] == 2
    assert report.summary["successful_optimizer_count"] >= 1
    assert report.summary["term_count"] == 1
    assert any(item.term == "x" for item in report.term_stability)
    assert set(report.model_statistics.columns) >= {
        "optimizer",
        "converged",
        "random_intercept_variance",
        "intraclass_correlation",
    }


def test_mixed_effects_robustness_dataframes() -> None:
    report = build_mixed_effects_robustness_report(
        make_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="group",
        optimizers=("lbfgs",),
    )

    assert not coefficient_comparison_to_dataframe(report).empty
    assert not stability_summary_to_dataframe(report).empty
    assert not model_comparison_to_dataframe(report).empty
    summary = robustness_summary_to_dataframe(report)
    assert "successful_optimizer_count" in summary["item"].tolist()
