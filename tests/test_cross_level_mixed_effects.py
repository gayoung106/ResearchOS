from __future__ import annotations

import numpy as np
import pandas as pd

from src.reporting.regression import write_korean_results_narrative
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.mixed_effects import fit_random_slope
from src.visualization.regression import build_regression_visualizations


def _data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(24), 16)
    z_group = rng.normal(size=24)
    z = np.repeat(z_group, 16)
    x = rng.normal(size=len(groups))
    ri = np.repeat(rng.normal(scale=0.7, size=24), 16)
    rs = np.repeat(rng.normal(scale=0.25, size=24), 16)
    y = (
        1.0
        + 0.7 * x
        + 0.4 * z
        + 0.55 * x * z
        + ri
        + rs * x
        + rng.normal(scale=0.7, size=len(groups))
    )
    return pd.DataFrame({"y": y, "x": x, "z": z, "group": groups})


def test_cross_level_interaction_estimation_and_metadata() -> None:
    df = _data()
    original = df.copy(deep=True)
    result = fit_random_slope(
        df,
        dependent_variable="y",
        independent_variables=["x", "z"],
        group_variable="group",
        random_slope_variable="x",
        cross_level_predictor="x",
        cross_level_moderator="z",
        level1_centering="group_mean",
        level2_centering="grand_mean",
        johnson_neyman=True,
    )
    pd.testing.assert_frame_equal(df, original)
    meta = result.metadata["cross_level_interaction"]
    assert result.model_type == "mixed_random_slope"
    assert meta["predictor_term"] == "x__group_mean_centered"
    assert meta["moderator_term"] == "z__grand_mean_centered"
    assert len(meta["conditional_effects"]) == 3
    assert meta["johnson_neyman"] is not None
    assert any(c.term == meta["interaction_term"] for c in result.coefficients)
    assert result.fit_statistics["cross_level_interaction_estimate"] > 0


def test_cross_level_reporting_effect_size_and_visualization(tmp_path) -> None:
    result = fit_random_slope(
        _data(7),
        dependent_variable="y",
        independent_variables=["x", "z"],
        group_variable="group",
        random_slope_variable="x",
        cross_level_predictor="x",
        cross_level_moderator="z",
        level1_centering="grand_mean",
        level2_centering="grand_mean",
    )
    effect = build_regression_effect_size_report(result)
    narrative = write_korean_results_narrative(result, effect)
    report = build_regression_visualizations(result, output_directory=tmp_path)
    assert "교차수준 상호작용" in narrative
    assert any(path.endswith("cross_level_interaction.png") for path in report.output_files)
    assert "marginal_r_squared" in effect.model_effects
