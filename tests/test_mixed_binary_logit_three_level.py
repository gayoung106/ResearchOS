import numpy as np
import pandas as pd
import pytest

from src.statistics.regression.mixed_binary_logit import (
    fit_mixed_binary_logit_three_level,
)
from src.statistics.regression.selector import fit_regression_by_level


def _data() -> pd.DataFrame:
    rng = np.random.default_rng(20260722)
    level3 = np.repeat(np.arange(6), 120)
    level2 = np.repeat(np.arange(24), 30)
    x = rng.normal(size=len(level2))
    region_effect = rng.normal(0, 0.45, 6)
    cluster_effect = rng.normal(0, 0.55, 24)
    linear = -0.35 + 0.9 * x + region_effect[level3] + cluster_effect[level2]
    y = rng.binomial(1, 1 / (1 + np.exp(-linear)))
    return pd.DataFrame({"y": y, "x": x, "cluster": level2, "region": level3})


def test_fit_three_level_mixed_binary_logit_returns_vpc() -> None:
    result = fit_mixed_binary_logit_three_level(
        _data(),
        dependent_variable="y",
        independent_variables=["x"],
        level2_group="cluster",
        level3_group="region",
    )

    assert result.model_type == "mixed_binary_logit_three_level"
    assert result.converged
    assert result.fit_statistics["level2_group_count"] == 24
    assert result.fit_statistics["level3_group_count"] == 6
    assert result.fit_statistics["level2_random_intercept_variance"] > 0
    assert result.fit_statistics["level3_random_intercept_variance"] > 0
    assert 0 < result.fit_statistics["icc"] < 1
    assert len(result.metadata["level2_random_effects"]) == 24
    assert len(result.metadata["level3_random_effects"]) == 6
    coefficient = next(item for item in result.coefficients if item.term == "x")
    assert coefficient.estimate > 0
    assert coefficient.exponentiated_estimate is not None


def test_selector_routes_three_level_mixed_binary_logit() -> None:
    result = fit_regression_by_level(
        _data(),
        dependent_variable="y",
        independent_variables=["x"],
        measurement_level="binary",
        model_type="mixed_binary_logit_three_level",
        mixed_effects_options={"level2_group": "cluster", "level3_group": "region"},
    )

    assert result.model_type == "mixed_binary_logit_three_level"


def test_three_level_mixed_binary_logit_rejects_cross_classified_groups() -> None:
    data = _data()
    data.loc[data.index[:5], "region"] = 5

    with pytest.raises(ValueError, match="중첩"):
        fit_mixed_binary_logit_three_level(
            data,
            dependent_variable="y",
            independent_variables=["x"],
            level2_group="cluster",
            level3_group="region",
        )
