"""Binary Logit 회귀모형의 예측성능·다중공선성 진단."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.statistics.diagnostics.ols import MulticollinearityResult
from src.statistics.regression.base import RegressionResult

_BINARY_DIAGNOSTIC_MODELS = {
    "binary_logit",
    "binary_cloglog",
    "binary_probit",
    "mixed_binary_logit_random_intercept",
    "mixed_binary_logit_random_slope",
    "mixed_binary_logit_three_level",
    "gee_logit",
}


@dataclass(slots=True)
class BinaryClassificationMetrics:
    """이항분류 성능지표."""

    threshold: float
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    accuracy: float
    sensitivity: float | None
    specificity: float | None
    precision: float | None
    negative_predictive_value: float | None
    f1_score: float | None
    roc_auc: float | None
    brier_score: float


@dataclass(slots=True)
class BinaryLogitDiagnosticsReport:
    """Binary Logit 진단 결과."""

    model_id: str
    model_type: str
    sample_size: int
    parameter_count: int
    event_count: int
    non_event_count: int
    events_per_parameter: float
    multicollinearity: list[MulticollinearityResult]
    classification_metrics: BinaryClassificationMetrics
    predictions: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _validate_binary_logit_result(
    result: RegressionResult,
) -> Any:
    """Binary Logit 결과인지 확인한다."""
    if result.model_type not in _BINARY_DIAGNOSTIC_MODELS:
        raise ValueError(
            "Binary Logit 진단은 model_type='binary_logit' 결과에만 적용할 수 있습니다."
        )

    if result.raw_result is None:
        raise ValueError("원본 statsmodels 결과 객체가 없습니다.")

    return result.raw_result


def _binary_diagnostic_arrays(
    result: RegressionResult,
) -> tuple[np.ndarray, np.ndarray, list[object], np.ndarray, list[str]]:
    fitted = _validate_binary_logit_result(result)
    diagnostics = result.metadata.get("diagnostics", {})

    if diagnostics:
        actual = np.asarray(diagnostics["endog"], dtype=int)
        probabilities = np.asarray(diagnostics["predicted_probability"], dtype=float)
        row_labels = list(diagnostics.get("row_labels", range(len(actual))))
        exog = np.asarray(diagnostics["exog"], dtype=float)
        exog_names = [str(name) for name in diagnostics["exog_names"]]
        return actual, probabilities, row_labels, exog, exog_names

    actual = np.asarray(fitted.model.endog, dtype=int)
    probabilities = np.asarray(fitted.predict(), dtype=float)
    row_labels = getattr(fitted.model.data, "row_labels", None)
    if row_labels is None:
        row_labels = list(range(len(actual)))
    exog = np.asarray(fitted.model.exog, dtype=float)
    exog_names = [str(name) for name in fitted.model.exog_names]
    return actual, probabilities, list(row_labels), exog, exog_names


def _safe_divide(
    numerator: float,
    denominator: float,
) -> float | None:
    """분모가 0이면 None을 반환한다."""
    if denominator == 0:
        return None

    return float(numerator / denominator)


def _expected_calibration_error(
    actual: np.ndarray,
    probabilities: np.ndarray,
    *,
    bin_count: int = 10,
) -> float:
    bins = np.linspace(0.0, 1.0, bin_count + 1)
    total = len(actual)
    error = 0.0

    for index in range(bin_count):
        lower = bins[index]
        upper = bins[index + 1]
        if index == bin_count - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)

        if not mask.any():
            continue

        observed_rate = float(np.mean(actual[mask]))
        predicted_rate = float(np.mean(probabilities[mask]))
        error += float(mask.sum() / total) * abs(observed_rate - predicted_rate)

    return float(error)


def _calculate_roc_auc(
    actual: np.ndarray,
    probabilities: np.ndarray,
) -> float | None:
    """순위합을 사용해 ROC-AUC를 계산한다."""
    event_mask = actual == 1
    non_event_mask = actual == 0

    event_count = int(event_mask.sum())
    non_event_count = int(non_event_mask.sum())

    if event_count == 0 or non_event_count == 0:
        return None

    probability_ranks = rankdata(
        probabilities,
        method="average",
    )
    event_rank_sum = float(probability_ranks[event_mask].sum())

    auc = (event_rank_sum - event_count * (event_count + 1) / 2) / (event_count * non_event_count)

    return float(auc)


def calculate_binary_multicollinearity(
    result: RegressionResult,
) -> list[MulticollinearityResult]:
    """Binary Logit 설계행렬의 VIF를 계산한다."""
    _, _, _, exog, exog_names = _binary_diagnostic_arrays(result)

    output: list[MulticollinearityResult] = []

    for index, variable_name in enumerate(exog_names):
        if variable_name.lower() in {
            "const",
            "intercept",
        }:
            continue

        try:
            vif = float(
                variance_inflation_factor(
                    exog,
                    index,
                )
            )
        except (
            ValueError,
            np.linalg.LinAlgError,
            ZeroDivisionError,
        ):
            vif = np.inf

        tolerance = 0.0 if (not np.isfinite(vif) or np.isclose(vif, 0.0)) else 1.0 / vif

        if not np.isfinite(vif) or vif >= 10:
            status = "FAIL"
            interpretation = "심각한 다중공선성이 의심됩니다."
        elif vif >= 5:
            status = "WARNING"
            interpretation = "다중공선성 주의가 필요합니다."
        else:
            status = "PASS"
            interpretation = "통상적인 VIF 기준을 충족합니다."

        output.append(
            MulticollinearityResult(
                variable_name=str(variable_name),
                vif=vif,
                tolerance=float(tolerance),
                status=status,
                interpretation=interpretation,
            )
        )

    return output


def calculate_binary_classification_metrics(
    result: RegressionResult,
    *,
    threshold: float = 0.5,
) -> tuple[
    BinaryClassificationMetrics,
    pd.DataFrame,
]:
    """Binary Logit의 분류지표와 사례별 예측값을 계산한다."""
    if not 0 < threshold < 1:
        raise ValueError("분류 임계값은 0과 1 사이여야 합니다.")

    actual, probabilities, row_labels, _, _ = _binary_diagnostic_arrays(result)
    predicted = (probabilities >= threshold).astype(int)

    true_positive = int(((actual == 1) & (predicted == 1)).sum())
    true_negative = int(((actual == 0) & (predicted == 0)).sum())
    false_positive = int(((actual == 0) & (predicted == 1)).sum())
    false_negative = int(((actual == 1) & (predicted == 0)).sum())

    sensitivity = _safe_divide(
        true_positive,
        true_positive + false_negative,
    )
    specificity = _safe_divide(
        true_negative,
        true_negative + false_positive,
    )
    precision = _safe_divide(
        true_positive,
        true_positive + false_positive,
    )
    negative_predictive_value = _safe_divide(
        true_negative,
        true_negative + false_negative,
    )

    if precision is None or sensitivity is None or precision + sensitivity == 0:
        f1_score = None
    else:
        f1_score = float(2 * precision * sensitivity / (precision + sensitivity))

    accuracy = float((true_positive + true_negative) / len(actual))
    roc_auc = _calculate_roc_auc(
        actual,
        probabilities,
    )
    brier_score = float(np.mean((probabilities - actual) ** 2))

    predictions = pd.DataFrame(
        {
            "row_index": row_labels,
            "actual": actual,
            "predicted_probability": probabilities,
            "predicted_class": predicted,
            "classification_correct": (actual == predicted),
            "absolute_prediction_error": np.abs(probabilities - actual),
        }
    )

    return (
        BinaryClassificationMetrics(
            threshold=threshold,
            true_positive=true_positive,
            true_negative=true_negative,
            false_positive=false_positive,
            false_negative=false_negative,
            accuracy=accuracy,
            sensitivity=sensitivity,
            specificity=specificity,
            precision=precision,
            negative_predictive_value=(negative_predictive_value),
            f1_score=f1_score,
            roc_auc=roc_auc,
            brier_score=brier_score,
        ),
        predictions,
    )


def build_binary_logit_diagnostics(
    result: RegressionResult,
    *,
    threshold: float = 0.5,
) -> BinaryLogitDiagnosticsReport:
    """Binary Logit 진단 보고서를 생성한다."""
    _validate_binary_logit_result(result)

    multicollinearity = calculate_binary_multicollinearity(result)
    (
        classification_metrics,
        predictions,
    ) = calculate_binary_classification_metrics(
        result,
        threshold=threshold,
    )

    sample_size = result.sample_size
    parameter_count = len(result.coefficients)

    actual = predictions["actual"].to_numpy(dtype=int)
    event_count = int((actual == 1).sum())
    non_event_count = int((actual == 0).sum())
    events_per_parameter = float(
        min(
            event_count,
            non_event_count,
        )
        / parameter_count
    )

    warnings: list[str] = []

    for item in multicollinearity:
        if item.status in {
            "WARNING",
            "FAIL",
        }:
            warnings.append(f"{item.variable_name}: {item.interpretation}")

    if events_per_parameter < 10:
        warnings.append("사건 또는 비사건 수 대비 추정 모수 수가 부족할 수 있습니다.")

    probabilities = predictions["predicted_probability"].to_numpy(dtype=float)
    calibration_mean_error = float(np.mean(actual - probabilities))
    expected_calibration_error = _expected_calibration_error(actual, probabilities)

    extreme_probability_count = int(((probabilities < 0.01) | (probabilities > 0.99)).sum())
    if extreme_probability_count:
        warnings.append(
            "0.01 미만 또는 0.99 초과의 "
            "극단 예측확률 사례가 "
            f"{extreme_probability_count}개 있습니다."
        )

    if classification_metrics.roc_auc is not None and classification_metrics.roc_auc < 0.7:
        warnings.append("ROC-AUC가 0.7 미만으로 판별력이 낮을 수 있습니다.")

    summary = {
        "model_id": result.model_id,
        "sample_size": sample_size,
        "parameter_count": parameter_count,
        "event_count": event_count,
        "non_event_count": non_event_count,
        "events_per_parameter": events_per_parameter,
        "vif_warning_count": sum(
            item.status
            in {
                "WARNING",
                "FAIL",
            }
            for item in multicollinearity
        ),
        "roc_auc": classification_metrics.roc_auc,
        "brier_score": (classification_metrics.brier_score),
        "calibration_mean_error": calibration_mean_error,
        "expected_calibration_error": expected_calibration_error,
        "accuracy": classification_metrics.accuracy,
        "sensitivity": (classification_metrics.sensitivity),
        "specificity": (classification_metrics.specificity),
        "extreme_probability_count": (extreme_probability_count),
    }

    return BinaryLogitDiagnosticsReport(
        model_id=result.model_id,
        model_type=result.model_type,
        sample_size=sample_size,
        parameter_count=parameter_count,
        event_count=event_count,
        non_event_count=non_event_count,
        events_per_parameter=events_per_parameter,
        multicollinearity=multicollinearity,
        classification_metrics=classification_metrics,
        predictions=predictions,
        warnings=warnings,
        summary=summary,
    )


def binary_multicollinearity_to_dataframe(
    report: BinaryLogitDiagnosticsReport,
) -> pd.DataFrame:
    """Binary Logit VIF 결과를 변환한다."""
    return pd.DataFrame([asdict(item) for item in report.multicollinearity])


def classification_metrics_to_dataframe(
    report: BinaryLogitDiagnosticsReport,
) -> pd.DataFrame:
    """분류 성능지표를 세로형 표로 변환한다."""
    values = asdict(report.classification_metrics)

    return pd.DataFrame(
        {
            "item": list(values.keys()),
            "value": list(values.values()),
        }
    )


def binary_predictions_to_dataframe(
    report: BinaryLogitDiagnosticsReport,
) -> pd.DataFrame:
    """사례별 Binary Logit 예측값을 반환한다."""
    return report.predictions.copy()


def binary_diagnostic_summary_to_dataframe(
    report: BinaryLogitDiagnosticsReport,
) -> pd.DataFrame:
    """Binary Logit 진단 요약을 변환한다."""
    values = {
        **report.summary,
        "warning_count": len(report.warnings),
    }

    return pd.DataFrame(
        {
            "item": list(values.keys()),
            "value": list(values.values()),
        }
    )
