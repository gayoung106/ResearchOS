import numpy as np
import pandas as pd

from src.statistics.regression.mixed_binary_logit import (
    fit_mixed_binary_logit_random_intercept,
)
from src.statistics.regression.selector import fit_regression_by_level


def _data() -> pd.DataFrame:
    rng = np.random.default_rng(20260720)
    group = np.repeat(np.arange(10), 30)
    x = rng.normal(size=len(group))
    random_intercept = rng.normal(0, 0.65, 10)
    probability = 1 / (1 + np.exp(-(-0.35 + 0.9 * x + random_intercept[group])))
    y = rng.binomial(1, probability)
    return pd.DataFrame({"y": y, "x": x, "cluster": group})


def test_fit_mixed_binary_logit_random_intercept_returns_odds_ratios() -> None:
    result = fit_mixed_binary_logit_random_intercept(
        _data(), dependent_variable="y", independent_variables=["x"], group_variable="cluster"
    )

    assert result.model_type == "mixed_binary_logit_random_intercept"
    assert result.converged
    assert result.fit_statistics["group_count"] == 10
    assert 0 <= result.fit_statistics["icc"] <= 1
    x = next(coef for coef in result.coefficients if coef.term == "x")
    assert x.exponentiated_estimate is not None
    assert x.estimate > 0
    assert len(result.metadata["random_effects"]) == 10


def test_selector_routes_explicit_mixed_binary_logit() -> None:
    result = fit_regression_by_level(
        _data(),
        dependent_variable="y",
        independent_variables=["x"],
        measurement_level="binary",
        model_type="mixed_binary_logit_random_intercept",
        group_variable="cluster",
    )
    assert result.model_type == "mixed_binary_logit_random_intercept"


def test_mixed_binary_logit_requires_two_groups() -> None:
    data = _data()
    data["cluster"] = 1
    try:
        fit_mixed_binary_logit_random_intercept(
            data, dependent_variable="y", independent_variables=["x"], group_variable="cluster"
        )
    except ValueError as error:
        assert "최소 2개 그룹" in str(error)
    else:
        raise AssertionError("ValueError가 발생해야 합니다.")
