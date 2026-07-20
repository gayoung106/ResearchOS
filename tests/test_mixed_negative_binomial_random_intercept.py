import numpy as np
import pandas as pd

from src.statistics.effects.regression import build_regression_effect_size_report
from src.statistics.regression.mixed_negative_binomial import (
    fit_mixed_negative_binomial_random_intercept,
)
from src.statistics.regression.selector import fit_regression_by_level


def _data() -> pd.DataFrame:
    rng = np.random.default_rng(20260725)
    groups = np.repeat(np.arange(10), 25)
    x = rng.normal(size=len(groups))
    random_intercept = rng.normal(0, 0.35, 10)
    mu = np.exp(0.4 + 0.45 * x + random_intercept[groups])
    alpha = 0.7
    shape = 1.0 / alpha
    rate = shape / mu
    latent_mean = rng.gamma(shape, 1.0 / rate)
    y = rng.poisson(latent_mean)
    return pd.DataFrame({"y": y, "x": x, "group": groups})


def test_fit_mixed_negative_binomial_random_intercept() -> None:
    result = fit_mixed_negative_binomial_random_intercept(
        _data(),
        dependent_variable="y",
        independent_variables=["x"],
        group_variable="group",
        quadrature_points=7,
        max_iterations=150,
    )

    assert result.model_type == "mixed_negative_binomial_random_intercept"
    assert result.fit_statistics["group_count"] == 10
    assert result.fit_statistics["dispersion_alpha"] > 0
    assert result.fit_statistics["random_intercept_variance"] > 0
    assert all(item.exponentiated_estimate is not None for item in result.coefficients)
    assert result.metadata["distribution"] == "negative_binomial_2"


def test_selector_and_effect_size_support_mixed_negative_binomial() -> None:
    result = fit_regression_by_level(
        _data(),
        dependent_variable="y",
        independent_variables=["x"],
        measurement_level="count",
        model_type="mixed_negative_binomial_random_intercept",
        group_variable="group",
        mixed_effects_options={"quadrature_points": 7, "max_iterations": 150},
    )
    report = build_regression_effect_size_report(result)

    assert result.model_type == "mixed_negative_binomial_random_intercept"
    assert any(item.effect_type == "incidence_rate_ratio" for item in report.effects)
