"""Diagnostics for multinomial logit models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.statistics.diagnostics.ols import MulticollinearityResult
from src.statistics.regression.base import RegressionResult


@dataclass(slots=True)
class MultinomialClassificationMetrics:
    accuracy: float
    mean_log_loss: float
    mean_maximum_predicted_probability: float


@dataclass(slots=True)
class MultinomialLogitDiagnosticsReport:
    model_id: str
    sample_size: int
    parameter_count: int
    category_count: int
    category_counts: dict[str, int]
    multicollinearity: list[MulticollinearityResult]
    classification_metrics: MultinomialClassificationMetrics
    predictions: pd.DataFrame
    confusion_matrix: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _validate_multinomial_result(result: RegressionResult) -> Any:
    if result.model_type != "multinomial_logit":
        raise ValueError("Multinomial diagnostics require model_type='multinomial_logit'.")
    if result.raw_result is None:
        raise ValueError("A fitted statsmodels result is required for multinomial diagnostics.")
    return result.raw_result


def calculate_multinomial_multicollinearity(
    result: RegressionResult,
) -> list[MulticollinearityResult]:
    fitted = _validate_multinomial_result(result)
    exog = np.asarray(fitted.model.exog, dtype=float)
    names = [str(name) for name in result.metadata.get("design_matrix_columns", [])]
    if len(names) != exog.shape[1]:
        names = [str(name) for name in getattr(fitted.model, "exog_names", [])]
    if len(names) != exog.shape[1]:
        names = [f"x{index + 1}" for index in range(exog.shape[1])]

    output: list[MulticollinearityResult] = []
    for index, name in enumerate(names):
        if name.lower() in {"const", "intercept"}:
            continue
        if exog.shape[1] == 1:
            vif = 1.0
        else:
            try:
                vif = float(variance_inflation_factor(exog, index))
            except (ValueError, IndexError, np.linalg.LinAlgError, ZeroDivisionError):
                vif = np.inf
        tolerance = 0.0 if not np.isfinite(vif) or np.isclose(vif, 0.0) else 1.0 / vif
        if not np.isfinite(vif) or vif >= 10:
            status = "FAIL"
            interpretation = "Severe multicollinearity is suspected."
        elif vif >= 5:
            status = "WARNING"
            interpretation = "Multicollinearity should be reviewed."
        else:
            status = "PASS"
            interpretation = "VIF is within the usual screening threshold."
        output.append(
            MulticollinearityResult(
                variable_name=name,
                vif=float(vif),
                tolerance=float(tolerance),
                status=status,
                interpretation=interpretation,
            )
        )
    return output


def calculate_multinomial_classification_metrics(
    result: RegressionResult,
) -> tuple[MultinomialClassificationMetrics, pd.DataFrame, pd.DataFrame]:
    fitted = _validate_multinomial_result(result)
    actual_codes = np.asarray(fitted.model.endog, dtype=int)
    probabilities = np.asarray(fitted.predict(), dtype=float)
    labels = [str(label) for label in result.metadata.get("category_labels", [])]
    if probabilities.ndim != 2 or probabilities.shape[0] != len(actual_codes):
        raise ValueError("Multinomial predicted probabilities have an unexpected shape.")
    if len(labels) != probabilities.shape[1]:
        labels = [str(index) for index in range(probabilities.shape[1])]

    predicted_codes = probabilities.argmax(axis=1)
    maximum_probabilities = probabilities.max(axis=1)
    clipped = np.clip(probabilities[np.arange(len(actual_codes)), actual_codes], 1e-15, 1.0)
    actual_labels = [labels[index] for index in actual_codes]
    predicted_labels = [labels[index] for index in predicted_codes]

    row_labels = result.metadata.get("row_labels") or list(range(len(actual_codes)))
    prediction_values: dict[str, Any] = {
        "row_index": row_labels,
        "actual": actual_labels,
        "predicted_category": predicted_labels,
        "maximum_predicted_probability": maximum_probabilities,
        "classification_correct": actual_codes == predicted_codes,
    }
    for index, label in enumerate(labels):
        prediction_values[f"probability_{label}"] = probabilities[:, index]

    predictions = pd.DataFrame(prediction_values)
    confusion_matrix = pd.crosstab(
        pd.Series(actual_labels, name="actual"),
        pd.Series(predicted_labels, name="predicted"),
        dropna=False,
    ).reindex(index=labels, columns=labels, fill_value=0)
    confusion_matrix.index.name = "actual"
    confusion_matrix.columns.name = "predicted"

    metrics = MultinomialClassificationMetrics(
        accuracy=float(np.mean(actual_codes == predicted_codes)),
        mean_log_loss=float(-np.mean(np.log(clipped))),
        mean_maximum_predicted_probability=float(np.mean(maximum_probabilities)),
    )
    return metrics, predictions, confusion_matrix


def build_multinomial_logit_diagnostics(
    result: RegressionResult,
) -> MultinomialLogitDiagnosticsReport:
    _validate_multinomial_result(result)
    multicollinearity = calculate_multinomial_multicollinearity(result)
    metrics, predictions, confusion_matrix = calculate_multinomial_classification_metrics(result)
    category_counts = {str(k): int(v) for k, v in result.metadata.get("category_counts", {}).items()}
    sample_size = result.sample_size
    parameter_count = int(result.fit_statistics.get("parameter_count", len(result.coefficients)))
    observations_per_parameter = float(sample_size / parameter_count) if parameter_count else np.nan
    minimum_category_count = min(category_counts.values()) if category_counts else 0

    warnings: list[str] = []
    for item in multicollinearity:
        if item.status in {"WARNING", "FAIL"}:
            warnings.append(f"{item.variable_name}: {item.interpretation}")
    if minimum_category_count < 10:
        warnings.append("At least one outcome category has fewer than 10 observations.")
    if np.isfinite(observations_per_parameter) and observations_per_parameter < 10:
        warnings.append("The sample size may be small relative to the number of multinomial parameters.")
    if metrics.mean_maximum_predicted_probability < 0.5:
        warnings.append("Mean maximum predicted probability is below 0.5.")

    summary = {
        "model_id": result.model_id,
        "sample_size": sample_size,
        "parameter_count": parameter_count,
        "category_count": int(result.fit_statistics.get("category_count", len(category_counts))),
        "minimum_category_count": minimum_category_count,
        "observations_per_parameter": observations_per_parameter,
        "vif_warning_count": sum(item.status in {"WARNING", "FAIL"} for item in multicollinearity),
        "accuracy": metrics.accuracy,
        "mean_log_loss": metrics.mean_log_loss,
        "mean_maximum_predicted_probability": metrics.mean_maximum_predicted_probability,
    }
    return MultinomialLogitDiagnosticsReport(
        model_id=result.model_id,
        sample_size=sample_size,
        parameter_count=parameter_count,
        category_count=summary["category_count"],
        category_counts=category_counts,
        multicollinearity=multicollinearity,
        classification_metrics=metrics,
        predictions=predictions,
        confusion_matrix=confusion_matrix,
        warnings=warnings,
        summary=summary,
    )


def multinomial_multicollinearity_to_dataframe(
    report: MultinomialLogitDiagnosticsReport,
) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.multicollinearity])


def multinomial_classification_metrics_to_dataframe(
    report: MultinomialLogitDiagnosticsReport,
) -> pd.DataFrame:
    values = asdict(report.classification_metrics)
    return pd.DataFrame({"item": list(values.keys()), "value": list(values.values())})


def multinomial_predictions_to_dataframe(
    report: MultinomialLogitDiagnosticsReport,
) -> pd.DataFrame:
    return report.predictions.copy()


def multinomial_confusion_matrix_to_dataframe(
    report: MultinomialLogitDiagnosticsReport,
) -> pd.DataFrame:
    return report.confusion_matrix.reset_index()


def multinomial_diagnostic_summary_to_dataframe(
    report: MultinomialLogitDiagnosticsReport,
) -> pd.DataFrame:
    values = {**report.summary, "warning_count": len(report.warnings)}
    return pd.DataFrame({"item": list(values.keys()), "value": list(values.values())})
