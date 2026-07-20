import numpy as np
import pandas as pd

from src.statistics.regression.mixed_binary_logit import (
    fit_mixed_binary_logit_random_slope,
)
from src.statistics.regression.selector import fit_regression_by_level


def _data() -> pd.DataFrame:
    rng = np.random.default_rng(20260721)
    group = np.repeat(np.arange(12), 30)
    x = rng.normal(size=len(group))
    random_intercept = rng.normal(0, 0.55, 12)
    random_slope = rng.normal(0, 0.35, 12)
    linear = -0.3 + (0.8 + random_slope[group]) * x + random_intercept[group]
    y = rng.binomial(1, 1 / (1 + np.exp(-linear)))
    return pd.DataFrame({"y": y, "x": x, "cluster": group})


def test_fit_mixed_binary_logit_random_slope_returns_variance_components() -> None:
    result = fit_mixed_binary_logit_random_slope(
        _data(),
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="cluster",
        random_slope_variable="x",
    )

    assert result.model_type == "mixed_binary_logit_random_slope"
    assert result.converged
    assert result.fit_statistics["group_count"] == 12
    assert result.fit_statistics["random_intercept_variance"] > 0
    assert result.fit_statistics["random_slope_variance"] > 0
    assert result.metadata["random_effect_covariance"] == "diagonal"
    assert len(result.metadata["random_effects"]) == 12
    assert len(result.metadata["random_slopes"]) == 12
    x = next(coefficient for coefficient in result.coefficients if coefficient.term == "x")
    assert x.exponentiated_estimate is not None
    assert x.estimate > 0


def test_selector_routes_mixed_binary_logit_random_slope() -> None:
    result = fit_regression_by_level(
        _data(),
        dependent_variable="y",
        independent_variables=["x"],
        measurement_level="binary",
        model_type="mixed_binary_logit_random_slope",
        group_variable="cluster",
        mixed_effects_options={"random_slope_variable": "x"},
    )

    assert result.model_type == "mixed_binary_logit_random_slope"


def test_mixed_binary_logit_random_slope_must_be_fixed_predictor() -> None:
    try:
        fit_mixed_binary_logit_random_slope(
            _data(),
            dependent_variable="y",
            independent_variables=["x"],
            group_variable="cluster",
            random_slope_variable="missing",
        )
    except ValueError as error:
        assert "독립변수" in str(error)
    else:
        raise AssertionError("ValueError가 발생해야 합니다.")
