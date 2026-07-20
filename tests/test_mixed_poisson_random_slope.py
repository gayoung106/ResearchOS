import numpy as np
import pandas as pd
import pytest

from src.statistics.regression.mixed_count import fit_mixed_poisson_random_slope
from src.statistics.regression.selector import fit_regression_by_level


def _data() -> pd.DataFrame:
    rng = np.random.default_rng(20260720)
    groups = np.repeat(["g1", "g2", "g3", "g4"], 18)
    x = rng.normal(size=len(groups))
    group_intercept = {"g1": -0.35, "g2": 0.1, "g3": 0.35, "g4": -0.05}
    group_slope = {"g1": 0.1, "g2": 0.35, "g3": 0.6, "g4": 0.8}
    eta = np.array(
        [
            0.45 + group_intercept[g] + (0.4 + group_slope[g]) * value
            for g, value in zip(groups, x, strict=True)
        ]
    )
    y = rng.poisson(np.exp(eta))
    return pd.DataFrame({"y": y, "x": x, "group": groups})


def test_fit_mixed_poisson_random_slope_returns_irr_and_variances() -> None:
    result = fit_mixed_poisson_random_slope(
        _data(),
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="group",
        random_slope_variable="x",
        max_iterations=300,
    )

    assert result.model_type == "mixed_poisson_random_slope"
    assert result.fit_statistics["group_count"] == 4
    assert result.fit_statistics["random_intercept_variance"] >= 0
    assert result.fit_statistics["random_slope_variance"] >= 0
    assert result.metadata["random_slope_variable"] == "x"
    assert result.metadata["random_effect_covariance"] == "diagonal"
    assert len(result.metadata["random_intercepts"]) == 4
    assert len(result.metadata["random_slopes"]) == 4
    assert all(item.exponentiated_estimate is not None for item in result.coefficients)


def test_selector_routes_mixed_poisson_random_slope() -> None:
    result = fit_regression_by_level(
        _data(),
        dependent_variable="y",
        independent_variables=["x"],
        measurement_level="count",
        model_type="mixed_poisson_random_slope",
        group_variable="group",
        mixed_effects_options={"random_slope_variable": "x", "max_iterations": 300},
    )

    assert result.model_type == "mixed_poisson_random_slope"
    assert result.metadata["random_slope_variable"] == "x"


def test_mixed_poisson_random_slope_requires_slope_in_predictors() -> None:
    with pytest.raises(ValueError, match="독립변수에 포함"):
        fit_mixed_poisson_random_slope(
            _data(),
            dependent_variable="y",
            independent_variables=[],
            group_variable="group",
            random_slope_variable="x",
        )
