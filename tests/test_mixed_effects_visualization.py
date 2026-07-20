"""Random Intercept 혼합효과모형 시각화 테스트."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.statistics.regression.mixed_effects import fit_random_intercept
from src.visualization.regression import build_regression_visualizations


def make_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(20260720)
    groups = np.repeat(np.arange(12), 8)
    x = rng.normal(size=len(groups))
    random_intercepts = rng.normal(scale=0.9, size=12)
    y = (
        1.2
        + 1.7 * x
        + random_intercepts[groups]
        + rng.normal(
            scale=0.45,
            size=len(groups),
        )
    )
    return pd.DataFrame({"y": y, "x": x, "group": groups})


def test_build_mixed_effects_visualizations(tmp_path: Path) -> None:
    result = fit_random_intercept(
        make_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="group",
        method="lbfgs",
        max_iterations=200,
    )

    report = build_regression_visualizations(
        result,
        output_directory=tmp_path,
    )

    expected_names = {
        "coefficient_forest.png",
        "residuals_vs_fitted.png",
        "residual_qq_plot.png",
        "random_intercepts.png",
    }

    assert report.model_type == "mixed_random_intercept"
    assert report.metadata["figure_count"] == 4
    assert report.warnings == []
    assert {Path(path).name for path in report.output_files} == expected_names
    assert all(Path(path).exists() for path in report.output_files)
    assert all(Path(path).stat().st_size > 0 for path in report.output_files)
