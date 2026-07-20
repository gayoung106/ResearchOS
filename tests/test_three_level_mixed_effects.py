from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.statistics.regression.mixed_effects import fit_three_level_mixed_effects
from src.statistics.regression.selector import fit_regression_by_level


def _three_level_data(seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for school in range(8):
        school_effect = rng.normal(0.0, 0.8)
        for classroom in range(4):
            class_effect = rng.normal(0.0, 0.55)
            class_id = f"s{school}_c{classroom}"
            for _ in range(10):
                x = rng.normal()
                y = 1.2 + 0.7 * x + school_effect + class_effect + rng.normal(0.0, 0.65)
                rows.append({"y": y, "x": x, "class": class_id, "school": f"s{school}"})
    return pd.DataFrame(rows)


def test_three_level_random_intercept_returns_variance_partition() -> None:
    result = fit_three_level_mixed_effects(
        _three_level_data(),
        dependent_variable="y",
        independent_variables=["x"],
        level2_group="class",
        level3_group="school",
    )
    assert result.model_type == "mixed_three_level"
    assert result.converged
    assert result.fit_statistics["level2_intercept_variance"] > 0
    assert result.fit_statistics["level3_intercept_variance"] > 0
    partition = result.fit_statistics["variance_partition"]
    assert sum(partition.values()) == pytest.approx(1.0)
    assert result.metadata["nested"] is True


def test_three_level_rejects_cross_classified_level2_groups() -> None:
    dataframe = _three_level_data()
    dataframe.loc[dataframe.index[:3], "class"] = "duplicated_class"
    dataframe.loc[dataframe.index[40:43], "class"] = "duplicated_class"
    with pytest.raises(ValueError, match="완전 중첩"):
        fit_three_level_mixed_effects(
            dataframe,
            dependent_variable="y",
            independent_variables=["x"],
            level2_group="class",
            level3_group="school",
        )


def test_selector_runs_three_level_model() -> None:
    result = fit_regression_by_level(
        _three_level_data(),
        dependent_variable="y",
        independent_variables=["x"],
        measurement_level="continuous",
        model_type="mixed_three_level",
        mixed_effects_options={"level2_group": "class", "level3_group": "school"},
    )
    assert result.model_type == "mixed_three_level"
    assert result.fit_statistics["level3_group_count"] == 8
