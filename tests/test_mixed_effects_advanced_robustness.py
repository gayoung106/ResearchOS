from __future__ import annotations

import numpy as np
import pandas as pd

from src.statistics.robustness.advanced_mixed_effects import (
    build_mixed_advanced_robustness_report,
    mixed_advanced_summary_to_dataframe,
    mixed_resampling_to_dataframe,
)


def make_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(20260720)
    groups = np.repeat(np.arange(10), 6)
    x = rng.normal(size=len(groups))
    random_intercepts = rng.normal(scale=0.7, size=10)
    y = 1.0 + 1.5 * x + random_intercepts[groups] + rng.normal(scale=0.4, size=len(groups))
    return pd.DataFrame({"y": y, "x": x, "group": groups})


def test_build_mixed_advanced_robustness_report() -> None:
    report = build_mixed_advanced_robustness_report(
        make_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="group",
        bootstrap_replications=20,
    )
    assert report.group_count == 10
    assert report.metadata["successful_bootstrap_replications"] >= 1
    assert any(item.term == "x" for item in report.coefficients)
    assert set(report.leave_one_group_out.columns) >= {
        "omitted_group",
        "term",
        "estimate",
        "intraclass_correlation",
    }
    assert not mixed_resampling_to_dataframe(report).empty
    assert "bootstrap_success_rate" in mixed_advanced_summary_to_dataframe(report)["item"].tolist()
