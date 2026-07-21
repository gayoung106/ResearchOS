"""Ordered Logit 회귀모형의 예측성능·다중공선성 진단."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.statistics.diagnostics.ols import MulticollinearityResult
from src.statistics.regression.base import RegressionResult


@dataclass(slots=True)
class OrderedClassificationMetrics:
    """순서형 분류 성능지표."""

    accuracy: float
    mean_absolute_category_error: float
    ranked_probability_score: float
    mean_maximum_predicted_probability: float


@dataclass(slots=True)
class OrderedLogitDiagnosticsReport:
    """Ordered Logit 진단 결과."""

    model_id: str
    model_type: str
    sample_size: int
    parameter_count: int
    category_count: int
    category_counts: dict[str, int]
    multicollinearity: list[MulticollinearityResult]
    classification_metrics: OrderedClassificationMetrics
    predictions: pd.DataFrame
    confusion_matrix: pd.DataFrame
    thresholds: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _validate_ordered_logit_result(
    result: RegressionResult,
) -> Any:
    """Validate that the result can use ordered-outcome diagnostics."""
    if result.model_type not in {"ordered_logit", "ordered_probit"}:
        raise ValueError(
            "Ordered diagnostics require model_type='ordered_logit' or 'ordered_probit'."
        )

    if result.raw_result is None:
        raise ValueError("A fitted statsmodels ordered result is required.")

    return result.raw_result

def _ordered_categories(
    actual: np.ndarray,
) -> list[float]:
    """실제 종속변수 범주를 오름차순으로 반환한다."""
    return sorted(np.unique(actual).astype(float).tolist())


def calculate_ordered_multicollinearity(
    result: RegressionResult,
) -> list[MulticollinearityResult]:
    """Ordered Logit 설계행렬의 VIF를 계산한다."""
    fitted = _validate_ordered_logit_result(result)

    exog = np.asarray(
        fitted.model.exog,
        dtype=float,
    )

    if exog.ndim != 2:
        raise ValueError("Ordered Logit 설계행렬이 2차원이 아닙니다.")

    predictor_count = int(exog.shape[1])

    raw_exog_names = list(
        getattr(
            fitted.model,
            "exog_names",
            [],
        )
    )

    design_matrix_columns = list(
        result.metadata.get(
            "design_matrix_columns",
            [],
        )
    )

    if len(design_matrix_columns) == predictor_count:
        predictor_names = [str(name) for name in design_matrix_columns]
    elif len(raw_exog_names) >= predictor_count:
        predictor_names = [str(name) for name in raw_exog_names[:predictor_count]]
    else:
        predictor_names = [f"x{index + 1}" for index in range(predictor_count)]

    output: list[MulticollinearityResult] = []

    for index in range(predictor_count):
        variable_name = predictor_names[index]

        if variable_name.lower() in {
            "const",
            "intercept",
        }:
            continue

        if predictor_count == 1:
            vif = 1.0
        else:
            try:
                vif = float(
                    variance_inflation_factor(
                        exog,
                        index,
                    )
                )
            except (
                ValueError,
                IndexError,
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
                variable_name=variable_name,
                vif=float(vif),
                tolerance=float(tolerance),
                status=status,
                interpretation=interpretation,
            )
        )

    return output


def _calculate_ranked_probability_score(
    actual: np.ndarray,
    probabilities: np.ndarray,
    categories: list[float],
) -> float:
    """순서형 예측의 Ranked Probability Score를 계산한다."""
    if len(categories) < 2:
        raise ValueError("Ranked Probability Score는 최소 2개 범주가 필요합니다.")

    cumulative_probabilities = np.cumsum(
        probabilities,
        axis=1,
    )[:, :-1]

    observed_cumulative = np.column_stack(
        [actual <= category for category in categories[:-1]]
    ).astype(float)

    row_scores = np.sum(
        (cumulative_probabilities - observed_cumulative) ** 2,
        axis=1,
    ) / (len(categories) - 1)

    return float(np.mean(row_scores))


def calculate_ordered_classification_metrics(
    result: RegressionResult,
) -> tuple[
    OrderedClassificationMetrics,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Ordered Logit 분류지표와 사례별 예측값을 계산한다."""
    fitted = _validate_ordered_logit_result(result)

    actual = np.asarray(
        fitted.model.endog,
        dtype=float,
    )
    probabilities = np.asarray(
        fitted.predict(),
        dtype=float,
    )
    categories = _ordered_categories(actual)

    if probabilities.ndim != 2:
        raise ValueError("Ordered Logit 예측확률 행렬이 2차원이 아닙니다.")

    if probabilities.shape[0] != len(actual):
        raise ValueError("예측확률 행 수와 실제 관측치 수가 일치하지 않습니다.")

    if probabilities.shape[1] != len(categories):
        raise ValueError("예측확률 열 수와 종속변수 범주 수가 일치하지 않습니다.")

    predicted_indices = probabilities.argmax(axis=1)
    predicted_categories = np.asarray(
        [categories[index] for index in predicted_indices],
        dtype=float,
    )
    maximum_probabilities = probabilities.max(axis=1)

    accuracy = float(np.mean(actual == predicted_categories))
    mean_absolute_category_error = float(np.mean(np.abs(actual - predicted_categories)))
    ranked_probability_score = _calculate_ranked_probability_score(
        actual,
        probabilities,
        categories,
    )
    mean_maximum_predicted_probability = float(np.mean(maximum_probabilities))

    row_labels = getattr(
        fitted.model.data,
        "row_labels",
        None,
    )
    if row_labels is None:
        row_labels = list(range(len(actual)))

    prediction_values: dict[str, Any] = {
        "row_index": row_labels,
        "actual": actual,
        "predicted_category": (predicted_categories),
        "maximum_predicted_probability": (maximum_probabilities),
        "classification_correct": (actual == predicted_categories),
        "absolute_category_error": np.abs(actual - predicted_categories),
    }

    for index, category in enumerate(categories):
        prediction_values[f"probability_{category:g}"] = probabilities[:, index]

    predictions = pd.DataFrame(prediction_values)

    confusion_matrix = pd.crosstab(
        pd.Series(
            actual,
            name="actual",
        ),
        pd.Series(
            predicted_categories,
            name="predicted",
        ),
        dropna=False,
    ).reindex(
        index=categories,
        columns=categories,
        fill_value=0,
    )

    confusion_matrix.index.name = "actual"
    confusion_matrix.columns.name = "predicted"

    return (
        OrderedClassificationMetrics(
            accuracy=accuracy,
            mean_absolute_category_error=(mean_absolute_category_error),
            ranked_probability_score=(ranked_probability_score),
            mean_maximum_predicted_probability=(mean_maximum_predicted_probability),
        ),
        predictions,
        confusion_matrix,
    )


def calculate_ordered_thresholds(
    result: RegressionResult,
) -> pd.DataFrame:
    """Ordered Logit의 변환된 임계값과 순서 상태를 반환한다."""
    fitted = _validate_ordered_logit_result(result)

    threshold_terms = list(
        result.metadata.get(
            "threshold_terms",
            [],
        )
    )

    transformed = np.asarray(
        fitted.model.transform_threshold_params(fitted.params),
        dtype=float,
    )

    finite_thresholds = transformed[1:-1]

    if len(finite_thresholds) != len(threshold_terms):
        raise ValueError("임계값 항목 수와 변환된 임계값 수가 일치하지 않습니다.")

    rows: list[dict[str, Any]] = []
    previous: float | None = None

    for term, transformed_value in zip(
        threshold_terms,
        finite_thresholds,
        strict=True,
    ):
        is_increasing = previous is None or transformed_value > previous

        rows.append(
            {
                "term": term,
                "raw_parameter": float(fitted.params[term]),
                "transformed_threshold": float(transformed_value),
                "strictly_increasing": bool(is_increasing),
            }
        )

        previous = float(transformed_value)

    return pd.DataFrame(
        rows,
        columns=[
            "term",
            "raw_parameter",
            "transformed_threshold",
            "strictly_increasing",
        ],
    )


def build_ordered_logit_diagnostics(
    result: RegressionResult,
) -> OrderedLogitDiagnosticsReport:
    """Ordered Logit 진단 보고서를 생성한다."""
    fitted = _validate_ordered_logit_result(result)

    multicollinearity = calculate_ordered_multicollinearity(result)

    (
        metrics,
        predictions,
        confusion_matrix,
    ) = calculate_ordered_classification_metrics(result)

    thresholds = calculate_ordered_thresholds(result)

    actual = np.asarray(
        fitted.model.endog,
        dtype=float,
    )
    categories = _ordered_categories(actual)

    category_counts = {f"{category:g}": int(np.sum(actual == category)) for category in categories}

    sample_size = int(fitted.nobs)
    parameter_count = int(len(fitted.params))
    observations_per_parameter = float(sample_size / parameter_count)
    minimum_category_count = min(category_counts.values())

    thresholds_increasing = (
        True if thresholds.empty else bool(thresholds["strictly_increasing"].all())
    )

    warnings: list[str] = []

    for item in multicollinearity:
        if item.status in {
            "WARNING",
            "FAIL",
        }:
            warnings.append(f"{item.variable_name}: {item.interpretation}")

    if minimum_category_count < 10:
        warnings.append("일부 종속변수 범주의 사례 수가 10개 미만입니다.")

    if observations_per_parameter < 10:
        warnings.append("표본 수 대비 추정 모수 수가 부족할 수 있습니다.")

    if not thresholds_increasing:
        warnings.append("변환된 Ordered Logit 임계값의 순서가 증가하지 않습니다.")

    if metrics.mean_maximum_predicted_probability < 0.5:
        warnings.append("평균 최대 예측확률이 0.5 미만입니다.")

    summary = {
        "model_id": result.model_id,
        "sample_size": sample_size,
        "parameter_count": (parameter_count),
        "category_count": len(categories),
        "minimum_category_count": (minimum_category_count),
        "observations_per_parameter": (observations_per_parameter),
        "vif_warning_count": sum(
            item.status
            in {
                "WARNING",
                "FAIL",
            }
            for item in multicollinearity
        ),
        "accuracy": metrics.accuracy,
        "mean_absolute_category_error": (metrics.mean_absolute_category_error),
        "ranked_probability_score": (metrics.ranked_probability_score),
        "mean_maximum_predicted_probability": (metrics.mean_maximum_predicted_probability),
        "thresholds_strictly_increasing": (thresholds_increasing),
    }

    return OrderedLogitDiagnosticsReport(
        model_id=result.model_id,
        model_type=result.model_type,
        sample_size=sample_size,
        parameter_count=parameter_count,
        category_count=len(categories),
        category_counts=category_counts,
        multicollinearity=multicollinearity,
        classification_metrics=metrics,
        predictions=predictions,
        confusion_matrix=confusion_matrix,
        thresholds=thresholds,
        warnings=warnings,
        summary=summary,
    )


def ordered_multicollinearity_to_dataframe(
    report: OrderedLogitDiagnosticsReport,
) -> pd.DataFrame:
    """Ordered Logit VIF 결과를 변환한다."""
    return pd.DataFrame([asdict(item) for item in report.multicollinearity])


def ordered_classification_metrics_to_dataframe(
    report: OrderedLogitDiagnosticsReport,
) -> pd.DataFrame:
    """Ordered Logit 분류 성능지표를 세로형 표로 변환한다."""
    values = asdict(report.classification_metrics)

    return pd.DataFrame(
        {
            "item": list(values.keys()),
            "value": list(values.values()),
        }
    )


def ordered_predictions_to_dataframe(
    report: OrderedLogitDiagnosticsReport,
) -> pd.DataFrame:
    """사례별 Ordered Logit 예측값을 반환한다."""
    return report.predictions.copy()


def ordered_confusion_matrix_to_dataframe(
    report: OrderedLogitDiagnosticsReport,
) -> pd.DataFrame:
    """Ordered Logit 혼동행렬을 반환한다."""
    return report.confusion_matrix.reset_index()


def ordered_thresholds_to_dataframe(
    report: OrderedLogitDiagnosticsReport,
) -> pd.DataFrame:
    """Ordered Logit 임계값 정보를 반환한다."""
    return report.thresholds.copy()


def ordered_diagnostic_summary_to_dataframe(
    report: OrderedLogitDiagnosticsReport,
) -> pd.DataFrame:
    """Ordered Logit 진단 요약을 세로형 표로 변환한다."""
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
