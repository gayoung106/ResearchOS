import numpy as np
import pandas as pd
import pytest

from src.statistics.regression.mixed_count import fit_mixed_poisson_three_level
from src.statistics.regression.selector import fit_regression_by_level


def _data() -> pd.DataFrame:
    rng = np.random.default_rng(20260724)
    region = np.repeat(np.arange(5), 120)
    cluster = np.repeat(np.arange(20), 30)
    x = rng.normal(size=len(cluster))
    region_effect = rng.normal(0, 0.25, 5)
    cluster_effect = rng.normal(0, 0.35, 20)
    linear = 0.35 + 0.45 * x + region_effect[region] + cluster_effect[cluster]
    y = rng.poisson(np.exp(linear))
    return pd.DataFrame({"y": y, "x": x, "cluster": cluster, "region": region})


def test_fit_three_level_mixed_poisson_returns_vpc_and_irr() -> None:
    result = fit_mixed_poisson_three_level(
        _data(),
        dependent_variable="y",
        independent_variables=["x"],
        level2_group="cluster",
        level3_group="region",
    )

    assert result.model_type == "mixed_poisson_three_level"
    assert result.fit_statistics["level2_group_count"] == 20
    assert result.fit_statistics["level3_group_count"] == 5
    assert result.fit_statistics["level2_random_intercept_variance"] > 0
    assert result.fit_statistics["level3_random_intercept_variance"] > 0
    assert result.fit_statistics["level2_vpc"] + result.fit_statistics[
        "level3_vpc"
    ] == pytest.approx(1.0)
    assert len(result.metadata["level2_random_effects"]) == 20
    assert len(result.metadata["level3_random_effects"]) == 5
    assert all(item.exponentiated_estimate is not None for item in result.coefficients)


def test_selector_routes_three_level_mixed_poisson() -> None:
    result = fit_regression_by_level(
        _data(),
        dependent_variable="y",
        independent_variables=["x"],
        measurement_level="count",
        model_type="mixed_poisson_three_level",
        mixed_effects_options={"level2_group": "cluster", "level3_group": "region"},
    )

    assert result.model_type == "mixed_poisson_three_level"


def test_three_level_mixed_poisson_rejects_cross_classified_groups() -> None:
    data = _data()
    data.loc[data.index[:5], "region"] = 4

    with pytest.raises(ValueError, match="중첩"):
        fit_mixed_poisson_three_level(
            data,
            dependent_variable="y",
            independent_variables=["x"],
            level2_group="cluster",
            level3_group="region",
        )
