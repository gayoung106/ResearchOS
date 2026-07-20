import numpy as np
import pandas as pd
import pytest

from src.statistics.diagnostics.count import build_count_diagnostics
from src.statistics.regression.mixed_count import fit_mixed_poisson_random_intercept
from src.statistics.regression.selector import fit_regression_by_level


def _data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    groups = np.repeat(np.arange(12), 20)
    x = rng.normal(size=len(groups))
    random_intercepts = rng.normal(0, 0.35, size=12)
    rate = np.exp(0.3 + 0.45 * x + random_intercepts[groups])
    return pd.DataFrame({"y": rng.poisson(rate), "x": x, "group": groups})


def test_fit_mixed_poisson_random_intercept_returns_irr() -> None:
    result = fit_mixed_poisson_random_intercept(
        _data(), dependent_variable="y", independent_variables=["x"], group_variable="group"
    )
    assert result.model_type == "mixed_poisson_random_intercept"
    assert result.fit_statistics["group_count"] == 12
    assert result.fit_statistics["random_intercept_variance"] >= 0
    assert all(coefficient.exponentiated_estimate > 0 for coefficient in result.coefficients)
    assert "diagnostics" in result.metadata

    diagnostics = build_count_diagnostics(result)
    assert diagnostics.model_type == result.model_type
    assert len(diagnostics.observations) == result.sample_size


def test_selector_routes_mixed_poisson_random_intercept() -> None:
    result = fit_regression_by_level(
        _data(),
        dependent_variable="y",
        independent_variables=["x"],
        measurement_level="count",
        model_type="mixed_poisson_random_intercept",
        group_variable="group",
    )
    assert result.model_type == "mixed_poisson_random_intercept"


def test_mixed_poisson_rejects_noninteger_outcome() -> None:
    data = _data()
    data["y"] = data["y"].astype(float) + 0.2
    with pytest.raises(ValueError, match="0 이상의 정수"):
        fit_mixed_poisson_random_intercept(
            data, dependent_variable="y", independent_variables=["x"], group_variable="group"
        )
