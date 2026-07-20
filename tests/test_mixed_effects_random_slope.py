from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config_models import AnalysisPlan, VariableMap
from src.reporting.regression import build_regression_publication_report
from src.statistics.diagnostics.mixed_effects import (
    build_mixed_effects_diagnostics,
    calculate_random_effects,
)
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.mixed_effects import fit_random_slope
from src.statistics.regression.selector import fit_regression_by_level
from src.visualization.regression import build_regression_visualizations
from tests.support.builders import build_regression_pipeline


def make_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260720)
    group_count, per_group = 18, 12
    groups = np.repeat(np.arange(group_count), per_group)
    x = rng.normal(size=len(groups))
    intercepts = rng.normal(0, 0.8, group_count)
    slopes = rng.normal(0, 0.45, group_count)
    y = 1.2 + 1.7 * x + intercepts[groups] + slopes[groups] * x + rng.normal(0, 0.35, len(groups))
    return pd.DataFrame({"y": y, "x": x, "group": groups})


def test_random_slope_fit_exposes_covariance_components() -> None:
    result = fit_random_slope(
        make_data(),
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="group",
        random_slope_variable="x",
    )
    assert result.model_type == "mixed_random_slope"
    assert result.converged
    assert result.fit_statistics["random_slope_variance"] > 0
    assert np.isfinite(result.fit_statistics["random_intercept_slope_covariance"])
    assert -1 <= result.fit_statistics["random_intercept_slope_correlation"] <= 1
    effects = calculate_random_effects(result)
    assert {"random_intercept", "random_slope"}.issubset(effects.columns)


def test_random_slope_diagnostics_effect_reporting_visualization(tmp_path: Path) -> None:
    result = fit_random_slope(
        make_data(),
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="group",
        random_slope_variable="x",
    )
    diagnostics = build_mixed_effects_diagnostics(result)
    assert "random_slope_variance" in diagnostics.summary
    assert "near_zero_slope_variance" in diagnostics.summary
    effects = build_regression_effect_size_report(result)
    assert effects.model_effects["random_slope_variance"] > 0
    report = build_regression_publication_report(result, effects)
    assert report.model_type == "mixed_random_slope"
    visual = build_regression_visualizations(result, output_directory=tmp_path)
    assert len(visual.output_files) >= 4


def test_selector_and_builder_register_random_slope(tmp_path: Path) -> None:
    result = fit_regression_by_level(
        make_data(),
        dependent_variable="y",
        independent_variables=["x"],
        measurement_level="continuous",
        model_type="mixed_random_slope",
        group_variable="group",
        mixed_effects_options={"random_slope_variable": "x"},
    )
    assert result.model_type == "mixed_random_slope"
    plan = AnalysisPlan.model_validate(
        {
            "variables": {"dependent": ["y"], "independent": ["x"]},
            "analyses": {
                "regression": {"enabled": True},
                "multilevel": {
                    "enabled": True,
                    "options": {"group_variable": "group", "random_slope_variable": "x"},
                },
            },
        }
    )
    variable_map = VariableMap.model_validate(
        {
            "variables": {
                "y": {"role": "dependent", "measurement_level": "continuous"},
                "x": {"role": "independent", "measurement_level": "continuous"},
                "group": {"role": "cluster", "measurement_level": "nominal"},
            }
        }
    )
    _, _, registration = build_regression_pipeline(
        tmp_path, analysis_plan=plan, variable_map=variable_map, project_name="random slope"
    )
    assert registration.registered
    assert registration.model_type == "mixed_random_slope"
