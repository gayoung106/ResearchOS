import numpy as np
import pandas as pd

from src.statistics.diagnostics.binary_logit import build_binary_logit_diagnostics
from src.statistics.regression.base import ModelCoefficient, RegressionResult


class _FakeMixedBinaryRawResult:
    pass


def test_mixed_binary_logit_diagnostics_accept_metadata_arrays() -> None:
    y = np.array([0, 1, 0, 1, 1, 0], dtype=int)
    x = np.column_stack([np.ones(len(y)), [-1.0, -0.4, 0.1, 0.3, 0.8, 1.2]])
    probability = np.array([0.2, 0.55, 0.35, 0.65, 0.75, 0.45])
    result = RegressionResult(
        model_id="main_model",
        model_type="mixed_binary_logit_random_intercept",
        dependent_variable="y",
        independent_variables=["x"],
        sample_size=len(y),
        coefficients=[
            ModelCoefficient("const", 0.0, 0.1, 0.0, 1.0, -0.2, 0.2, 1.0),
            ModelCoefficient("x", 0.7, 0.2, 3.5, 0.001, 0.3, 1.1, 2.01),
        ],
        fit_statistics={"group_count": 2, "event_count": 3, "non_event_count": 3},
        converged=True,
        standard_error_type="variational_bayes_posterior_sd",
        metadata={
            "diagnostics": {
                "endog": y.tolist(),
                "predicted_probability": probability.tolist(),
                "exog": x.tolist(),
                "exog_names": ["const", "x"],
                "row_labels": list(range(len(y))),
            }
        },
        raw_result=_FakeMixedBinaryRawResult(),
    )

    report = build_binary_logit_diagnostics(result)

    assert report.model_id == "main_model"
    assert report.sample_size == len(y)
    assert report.event_count == 3
    assert report.classification_metrics.accuracy >= 0
    assert 0 <= report.summary["expected_calibration_error"] <= 1
    assert "calibration_mean_error" in report.summary
    assert isinstance(report.predictions, pd.DataFrame)
