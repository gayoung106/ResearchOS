"""Poisson 및 Negative Binomial 계수형 회귀 진단."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.statistics.diagnostics.ols import MulticollinearityResult
from src.statistics.regression.base import RegressionResult

_COUNT_DIAGNOSTIC_MODELS = {
    "poisson",
    "negative_binomial",
    "zero_inflated_poisson",
    "zero_inflated_negative_binomial",
    "mixed_poisson_random_intercept",
    "mixed_poisson_random_slope",
    "mixed_poisson_three_level",
    "mixed_negative_binomial_random_intercept",
    "mixed_negative_binomial_random_slope",
    "mixed_negative_binomial_three_level",
}

_MIXED_NEGATIVE_BINOMIAL_MODELS = {
    "mixed_negative_binomial_random_intercept",
    "mixed_negative_binomial_random_slope",
    "mixed_negative_binomial_three_level",
}


@dataclass(slots=True)
class CountPredictionMetrics:
    """계수형 회귀 예측성능 지표."""

    mean_absolute_error: float
    root_mean_squared_error: float
    mean_error: float
    observed_mean: float
    predicted_mean: float
    observed_zero_proportion: float
    predicted_zero_proportion: float
    zero_proportion_difference: float


@dataclass(slots=True)
class CountDiagnosticsReport:
    """계수형 회귀 진단 결과."""

    model_id: str
    model_type: str
    sample_size: int
    parameter_count: int
    multicollinearity: list[MulticollinearityResult]
    prediction_metrics: CountPredictionMetrics
    observations: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _validate_count_result(
    result: RegressionResult,
) -> Any:
    """지원되는 계수형 회귀결과인지 확인한다."""
    if result.model_type not in _COUNT_DIAGNOSTIC_MODELS:
        raise ValueError(
            "계수형 회귀 진단은 Poisson 또는 Negative Binomial 결과에만 적용할 수 있습니다."
        )

    if result.raw_result is None:
        raise ValueError("원본 statsmodels 결과 객체가 없습니다.")

    return result.raw_result


def _count_diagnostic_arrays(
    result: RegressionResult,
) -> tuple[np.ndarray, np.ndarray, list[object], np.ndarray, list[str]]:
    fitted = _validate_count_result(result)
    diagnostics = result.metadata.get("diagnostics", {})

    if diagnostics:
        actual = np.asarray(diagnostics["endog"], dtype=float)
        predicted = np.asarray(diagnostics["predicted_mean"], dtype=float)
        row_labels = list(diagnostics.get("row_labels", range(len(actual))))
        exog = np.asarray(diagnostics["exog"], dtype=float)
        exog_names = [str(name) for name in diagnostics["exog_names"]]
        return actual, predicted, row_labels, exog, exog_names

    actual = np.asarray(fitted.model.endog, dtype=float)
    predicted = np.asarray(fitted.predict(), dtype=float)
    row_labels = getattr(fitted.model.data, "row_labels", None)
    if row_labels is None:
        row_labels = list(range(len(actual)))
    exog = np.asarray(fitted.model.exog, dtype=float)
    raw_names = list(getattr(fitted.model, "exog_names", []))
    predictor_count = int(exog.shape[1])
    if len(raw_names) >= predictor_count:
        exog_names = [str(name) for name in raw_names[:predictor_count]]
    else:
        exog_names = [f"x{index + 1}" for index in range(predictor_count)]
    return actual, predicted, list(row_labels), exog, exog_names


def _design_matrix(
    result: RegressionResult,
) -> tuple[np.ndarray, list[str]]:
    """Return the design matrix and column names used for diagnostics."""
    _, _, _, exog, names = _count_diagnostic_arrays(result)

    if exog.ndim != 2:
        raise ValueError("count regression design matrix must be two-dimensional.")

    return exog, names


def calculate_count_multicollinearity(
    result: RegressionResult,
) -> list[MulticollinearityResult]:
    """계수형 회귀 설계행렬의 VIF와 tolerance를 계산한다."""
    exog, names = _design_matrix(result)
    predictor_count = int(exog.shape[1])
    output: list[MulticollinearityResult] = []

    for index in range(predictor_count):
        variable_name = names[index]

        if variable_name.lower() in {
            "const",
            "intercept",
        }:
            continue

        if predictor_count <= 2:
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


def _nb2_alpha(
    result: RegressionResult,
) -> float:
    """Negative Binomial 과산포 모수를 반환한다."""
    if result.model_type not in {
        "negative_binomial",
        "zero_inflated_negative_binomial",
        *_MIXED_NEGATIVE_BINOMIAL_MODELS,
    }:
        return 0.0

    alpha = result.fit_statistics.get("alpha", result.fit_statistics.get("dispersion_alpha"))
    if alpha is None:
        return 0.0

    return max(float(alpha), 0.0)


def _pearson_residuals(
    actual: np.ndarray,
    predicted: np.ndarray,
    alpha: float,
) -> np.ndarray:
    variance = predicted if alpha <= 0 else predicted + alpha * predicted**2
    denominator = np.sqrt(
        np.maximum(
            variance,
            1e-12,
        )
    )
    return (actual - predicted) / denominator


def _poisson_deviance_residuals(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    predicted = np.maximum(
        predicted,
        1e-12,
    )
    logarithmic = np.zeros_like(
        actual,
        dtype=float,
    )
    positive = actual > 0
    logarithmic[positive] = actual[positive] * np.log(actual[positive] / predicted[positive])
    deviance = 2 * (logarithmic - (actual - predicted))
    return np.sign(actual - predicted) * np.sqrt(
        np.maximum(
            deviance,
            0.0,
        )
    )


def _negative_binomial_deviance_residuals(
    actual: np.ndarray,
    predicted: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if alpha <= 0:
        return _poisson_deviance_residuals(
            actual,
            predicted,
        )

    predicted = np.maximum(
        predicted,
        1e-12,
    )
    first = np.zeros_like(
        actual,
        dtype=float,
    )
    positive = actual > 0
    first[positive] = actual[positive] * np.log(actual[positive] / predicted[positive])

    second = (actual + 1 / alpha) * np.log((1 + alpha * actual) / (1 + alpha * predicted))

    deviance = 2 * (first - second)
    return np.sign(actual - predicted) * np.sqrt(
        np.maximum(
            deviance,
            0.0,
        )
    )


def _leverage(
    exog: np.ndarray,
) -> np.ndarray:
    """설계행렬 기반 근사 leverage를 계산한다."""
    projection = exog @ np.linalg.pinv(exog)
    return np.clip(
        np.diag(projection),
        0.0,
        1.0,
    )


def calculate_count_predictions(
    result: RegressionResult,
) -> tuple[
    CountPredictionMetrics,
    pd.DataFrame,
]:
    """예측성능과 사례별 잔차·영향력 지표를 계산한다."""
    fitted = _validate_count_result(result)
    actual, predicted, row_labels, _, _ = _count_diagnostic_arrays(result)

    if actual.shape[0] != predicted.shape[0]:
        raise ValueError("실제 관측치와 예측값의 길이가 일치하지 않습니다.")

    predicted = np.maximum(
        predicted,
        1e-12,
    )
    alpha = _nb2_alpha(result)
    raw_residual = actual - predicted
    if result.model_type in {
        "zero_inflated_poisson",
        "zero_inflated_negative_binomial",
    }:
        predicted_variance = np.asarray(
            fitted.predict(which="var"),
            dtype=float,
        )
        pearson_residual = (actual - predicted) / np.sqrt(np.maximum(predicted_variance, 1e-12))
    else:
        pearson_residual = _pearson_residuals(
            actual,
            predicted,
            alpha,
        )

    if result.model_type in {
        "negative_binomial",
        "zero_inflated_negative_binomial",
        *_MIXED_NEGATIVE_BINOMIAL_MODELS,
    }:
        deviance_residual = _negative_binomial_deviance_residuals(
            actual,
            predicted,
            alpha,
        )
    else:
        deviance_residual = _poisson_deviance_residuals(
            actual,
            predicted,
        )

    exog, _ = _design_matrix(result)
    leverage = _leverage(exog)
    parameter_count = int(exog.shape[1])
    sample_size = len(actual)
    leverage_threshold = 2 * parameter_count / sample_size
    residual_flag = np.abs(pearson_residual) > 3
    leverage_flag = leverage > leverage_threshold

    observed_zero_proportion = float(np.mean(actual == 0))

    if result.model_type in {
        "zero_inflated_poisson",
        "zero_inflated_negative_binomial",
    }:
        predicted_zero_probability = np.asarray(
            fitted.predict(which="prob-zero"),
            dtype=float,
        )
    elif result.model_type == "negative_binomial" or result.model_type in _MIXED_NEGATIVE_BINOMIAL_MODELS:
        predicted_zero_probability = (1 + alpha * predicted) ** (-1 / alpha)
    else:
        predicted_zero_probability = np.exp(-predicted)

    predicted_zero_proportion = float(np.mean(predicted_zero_probability))

    metrics = CountPredictionMetrics(
        mean_absolute_error=float(np.mean(np.abs(raw_residual))),
        root_mean_squared_error=float(np.sqrt(np.mean(raw_residual**2))),
        mean_error=float(np.mean(raw_residual)),
        observed_mean=float(np.mean(actual)),
        predicted_mean=float(np.mean(predicted)),
        observed_zero_proportion=(observed_zero_proportion),
        predicted_zero_proportion=(predicted_zero_proportion),
        zero_proportion_difference=float(observed_zero_proportion - predicted_zero_proportion),
    )

    observations = pd.DataFrame(
        {
            "row_index": row_labels,
            "actual": actual,
            "predicted": predicted,
            "raw_residual": raw_residual,
            "pearson_residual": (pearson_residual),
            "deviance_residual": (deviance_residual),
            "absolute_error": np.abs(raw_residual),
            "squared_error": (raw_residual**2),
            "predicted_zero_probability": (predicted_zero_probability),
            "leverage": leverage,
            "pearson_residual_flag": (residual_flag),
            "leverage_flag": (leverage_flag),
            "any_diagnostic_flag": (residual_flag | leverage_flag),
        }
    )

    return metrics, observations


def build_count_diagnostics(
    result: RegressionResult,
) -> CountDiagnosticsReport:
    """계수형 회귀 진단 보고서를 생성한다."""
    _validate_count_result(result)
    multicollinearity = calculate_count_multicollinearity(result)
    metrics, observations = calculate_count_predictions(result)

    sample_size = result.sample_size
    parameter_count = len(result.coefficients)
    flagged_count = int(observations["any_diagnostic_flag"].sum())
    extreme_residual_count = int(observations["pearson_residual_flag"].sum())
    high_leverage_count = int(observations["leverage_flag"].sum())
    residual_degrees_of_freedom = max(sample_size - parameter_count, 1)
    pearson_dispersion_ratio = float(
        np.sum(observations["pearson_residual"].to_numpy(dtype=float) ** 2)
        / residual_degrees_of_freedom
    )

    warnings: list[str] = []

    for item in multicollinearity:
        if item.status in {
            "WARNING",
            "FAIL",
        }:
            warnings.append(f"{item.variable_name}: {item.interpretation}")

    zero_difference = abs(metrics.zero_proportion_difference)
    if zero_difference > 0.1:
        warnings.append(
            "관측 0 비율과 모형의 평균 예측 0 비율 차이가 "
            "0.10을 초과합니다. 영과잉 모형을 검토하세요."
        )

    if extreme_residual_count > 0:
        warnings.append(f"절댓값 3을 초과하는 Pearson 잔차가 {extreme_residual_count}개 있습니다.")

    if high_leverage_count > 0:
        warnings.append(f"높은 leverage로 표시된 관측치가 {high_leverage_count}개 있습니다.")

    if result.model_type in {
        "poisson",
        "zero_inflated_poisson",
    }:
        dispersion_ratio = result.fit_statistics.get("dispersion_ratio")
        if (
            dispersion_ratio is not None
            and np.isfinite(float(dispersion_ratio))
            and float(dispersion_ratio) > 1.5
        ):
            warnings.append("Poisson Pearson 분산비가 1.5를 초과합니다.")

    summary = {
        "model_id": result.model_id,
        "model_type": result.model_type,
        "sample_size": sample_size,
        "parameter_count": parameter_count,
        "residual_degrees_of_freedom": residual_degrees_of_freedom,
        "pearson_dispersion_ratio": pearson_dispersion_ratio,
        "mean_absolute_error": (metrics.mean_absolute_error),
        "root_mean_squared_error": (metrics.root_mean_squared_error),
        "mean_error": metrics.mean_error,
        "observed_mean": metrics.observed_mean,
        "predicted_mean": metrics.predicted_mean,
        "observed_zero_proportion": (metrics.observed_zero_proportion),
        "predicted_zero_proportion": (metrics.predicted_zero_proportion),
        "zero_proportion_difference": (metrics.zero_proportion_difference),
        "extreme_pearson_residual_count": (extreme_residual_count),
        "high_leverage_count": (high_leverage_count),
        "flagged_observation_count": (flagged_count),
    }

    if result.model_type == "poisson":
        summary["dispersion_ratio"] = result.fit_statistics.get("dispersion_ratio")
    elif result.model_type in {
        "negative_binomial",
        "zero_inflated_negative_binomial",
        *_MIXED_NEGATIVE_BINOMIAL_MODELS,
    }:
        summary["alpha"] = result.fit_statistics.get(
            "alpha", result.fit_statistics.get("dispersion_alpha")
        )
    else:
        summary["inflation_model"] = result.metadata.get("inflation_model")

    return CountDiagnosticsReport(
        model_id=result.model_id,
        model_type=result.model_type,
        sample_size=sample_size,
        parameter_count=parameter_count,
        multicollinearity=multicollinearity,
        prediction_metrics=metrics,
        observations=observations,
        warnings=warnings,
        summary=summary,
    )


def count_multicollinearity_to_dataframe(
    report: CountDiagnosticsReport,
) -> pd.DataFrame:
    """VIF 결과를 데이터프레임으로 변환한다."""
    return pd.DataFrame([asdict(item) for item in report.multicollinearity])


def count_prediction_metrics_to_dataframe(
    report: CountDiagnosticsReport,
) -> pd.DataFrame:
    """예측성능 지표를 세로형 표로 변환한다."""
    values = asdict(report.prediction_metrics)
    return pd.DataFrame(
        {
            "item": list(values.keys()),
            "value": list(values.values()),
        }
    )


def count_observations_to_dataframe(
    report: CountDiagnosticsReport,
) -> pd.DataFrame:
    """사례별 계수형 회귀 진단값을 반환한다."""
    return report.observations.copy()


def count_diagnostic_summary_to_dataframe(
    report: CountDiagnosticsReport,
) -> pd.DataFrame:
    """계수형 회귀 진단 요약을 세로형 표로 변환한다."""
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
