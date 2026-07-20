import numpy as np
import pandas as pd

from src.statistics.diagnostics.mixed_effects import build_mixed_effects_diagnostics
from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.mixed_effects import fit_multiple_random_slopes
from src.statistics.regression.selector import fit_regression_by_level


def make_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(24), 12)
    x1 = rng.normal(size=len(groups))
    x2 = rng.normal(size=len(groups))
    ri = rng.normal(0, 0.7, 24)[groups]
    rs1 = rng.normal(0, 0.25, 24)[groups]
    rs2 = rng.normal(0, 0.20, 24)[groups]
    y = 1.0 + 0.8 * x1 - 0.5 * x2 + ri + rs1 * x1 + rs2 * x2 + rng.normal(0, 0.7, len(groups))
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "group": groups})


def test_multiple_random_slopes_fit_and_diagnostics():
    result = fit_multiple_random_slopes(
        make_data(),
        dependent_variable="y",
        independent_variables=["x1", "x2"],
        group_variable="group",
        random_slope_variables=["x1", "x2"],
        max_iterations=500,
    )
    assert result.model_type == "mixed_random_slope"
    assert result.metadata["random_slope_variables"] == ["x1", "x2"]
    assert set(result.fit_statistics["random_slope_variances"]) == {"x1", "x2"}
    assert set(result.fit_statistics["random_effect_correlation_matrix"]) == {
        "intercept",
        "x1",
        "x2",
    }
    diagnostics = build_mixed_effects_diagnostics(result)
    assert {"random_slope__x1", "random_slope__x2"}.issubset(diagnostics.random_effects.columns)
    effects = build_regression_effect_size_report(result)
    assert (
        effects.model_effects["conditional_r_squared"]
        >= effects.model_effects["marginal_r_squared"]
    )


def test_selector_accepts_multiple_random_slopes():
    result = fit_regression_by_level(
        make_data(),
        dependent_variable="y",
        independent_variables=["x1", "x2"],
        measurement_level="continuous",
        model_type="mixed_random_slope",
        group_variable="group",
        mixed_effects_options={"random_slope_variables": ["x1", "x2"], "max_iterations": 500},
    )
    assert result.metadata["random_slope_variables"] == ["x1", "x2"]
