from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.statistics.regression.mixed_effects import fit_multiple_random_slopes, fit_random_slope


def make_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260720)
    group_count = 30
    n_per_group = 10
    groups = np.repeat(np.arange(group_count), n_per_group)
    x1 = rng.normal(size=len(groups))
    x2 = rng.normal(size=len(groups))
    ri = rng.normal(0, 0.9, group_count)
    rs1 = rng.normal(0, 0.35, group_count)
    rs2 = rng.normal(0, 0.25, group_count)
    y = (
        1.0
        + 1.4 * x1
        - 0.8 * x2
        + ri[groups]
        + rs1[groups] * x1
        + rs2[groups] * x2
        + rng.normal(0, 0.5, len(groups))
    )
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "group": groups})


def test_uncorrelated_single_random_slope_constrains_covariance() -> None:
    result = fit_random_slope(
        make_data(),
        dependent_variable="y",
        independent_variables=["x1", "x2"],
        group_variable="group",
        random_slope_variable="x1",
        random_effect_covariance="diagonal",
    )
    assert result.converged
    assert result.metadata["random_effect_covariance"] == "diagonal"
    assert result.metadata["random_effects_correlated"] is False
    assert result.fit_statistics["random_intercept_slope_covariance"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert result.fit_statistics["random_intercept_slope_correlation"] == pytest.approx(
        0.0, abs=1e-12
    )


def test_uncorrelated_multiple_random_slopes_constrains_all_off_diagonals() -> None:
    result = fit_multiple_random_slopes(
        make_data(),
        dependent_variable="y",
        independent_variables=["x1", "x2"],
        group_variable="group",
        random_slope_variables=["x1", "x2"],
        random_effect_covariance="diagonal",
    )
    matrix = result.fit_statistics["random_effect_correlation_matrix"]
    assert result.converged
    assert matrix["intercept"]["x1"] == pytest.approx(0.0, abs=1e-12)
    assert matrix["intercept"]["x2"] == pytest.approx(0.0, abs=1e-12)
    assert matrix["x1"]["x2"] == pytest.approx(0.0, abs=1e-12)


def test_rejects_unknown_random_effect_covariance() -> None:
    with pytest.raises(ValueError, match="correlated 또는 diagonal"):
        fit_random_slope(
            make_data(),
            dependent_variable="y",
            independent_variables=["x1"],
            group_variable="group",
            random_slope_variable="x1",
            random_effect_covariance="unknown",
        )
