"""OLS 부트스트랩·잭나이프·군집강건 추정 모듈."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.statistics.regression.base import prepare_model_data


@dataclass(slots=True)
class ResampledCoefficient:
    """재표집 또는 군집강건 계수 결과."""

    term: str
    method: str
    estimate: float
    standard_error: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    p_value: float | None
    successful_replications: int | None = None
    requested_replications: int | None = None


@dataclass(slots=True)
class ResamplingReport:
    """재표집 강건성 분석 결과."""

    method: str
    model_id: str
    sample_size: int
    coefficients: list[ResampledCoefficient]
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _prepare_ols_arrays(
    dataframe: pd.DataFrame,
    dependent_variable: str,
    independent_variables: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    model_data = prepare_model_data(
        dataframe,
        dependent_variable,
        independent_variables,
    )
    outcome = model_data[dependent_variable]
    predictors = sm.add_constant(
        model_data[independent_variables],
        has_constant="add",
    )
    return model_data, outcome, predictors


def fit_cluster_robust_ols(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    cluster_variable: str,
    model_id: str = "main_model",
) -> ResamplingReport:
    """군집강건 표준오차를 적용한 OLS를 적합한다."""
    if cluster_variable not in dataframe.columns:
        raise KeyError(f"군집변수가 데이터에 없습니다: {cluster_variable}")

    required = [
        dependent_variable,
        *independent_variables,
        cluster_variable,
    ]
    model_data = dataframe[required].copy()

    for column in [dependent_variable, *independent_variables]:
        model_data[column] = pd.to_numeric(
            model_data[column],
            errors="coerce",
        )

    model_data = model_data.dropna()

    if model_data.empty:
        raise ValueError("군집강건 분석에 사용할 완전사례가 없습니다.")

    cluster_count = int(model_data[cluster_variable].nunique(dropna=True))
    if cluster_count < 2:
        raise ValueError("군집은 최소 2개 이상이어야 합니다.")

    outcome = model_data[dependent_variable]
    predictors = sm.add_constant(
        model_data[independent_variables],
        has_constant="add",
    )

    fitted = sm.OLS(outcome, predictors).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": model_data[cluster_variable],
            "use_correction": True,
        },
    )

    confidence_intervals = fitted.conf_int()
    coefficients = [
        ResampledCoefficient(
            term=str(term),
            method="cluster_robust",
            estimate=float(fitted.params[term]),
            standard_error=float(fitted.bse[term]),
            confidence_interval_lower=float(confidence_intervals.loc[term, 0]),
            confidence_interval_upper=float(confidence_intervals.loc[term, 1]),
            p_value=float(fitted.pvalues[term]),
        )
        for term in fitted.params.index
    ]

    warnings: list[str] = []
    if cluster_count < 30:
        warnings.append(f"군집 수가 {cluster_count}개로 적어 군집강건 추론이 불안정할 수 있습니다.")

    return ResamplingReport(
        method="cluster_robust",
        model_id=model_id,
        sample_size=int(fitted.nobs),
        coefficients=coefficients,
        warnings=warnings,
        metadata={
            "cluster_variable": cluster_variable,
            "cluster_count": cluster_count,
            "dropped_case_count": len(dataframe) - len(model_data),
        },
    )


def bootstrap_ols(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    model_id: str = "main_model",
    replications: int = 2000,
    confidence_level: float = 0.95,
    random_seed: int = 2026,
) -> ResamplingReport:
    """사례 단위 비모수 부트스트랩 OLS를 수행한다."""
    if replications < 100:
        raise ValueError("부트스트랩 반복 수는 최소 100이어야 합니다.")

    model_data, outcome, predictors = _prepare_ols_arrays(
        dataframe,
        dependent_variable,
        independent_variables,
    )

    baseline = sm.OLS(outcome, predictors).fit()
    terms = list(baseline.params.index)
    rng = np.random.default_rng(random_seed)
    estimates: list[np.ndarray] = []
    failed = 0

    for _ in range(replications):
        positions = rng.integers(
            0,
            len(model_data),
            size=len(model_data),
        )
        sampled_y = outcome.iloc[positions].reset_index(drop=True)
        sampled_x = predictors.iloc[positions].reset_index(drop=True)

        try:
            fitted = sm.OLS(sampled_y, sampled_x).fit()
            estimates.append(fitted.params.reindex(terms).to_numpy(dtype=float))
        except (ValueError, np.linalg.LinAlgError):
            failed += 1

    if not estimates:
        raise RuntimeError("모든 부트스트랩 반복이 실패했습니다.")

    matrix = np.vstack(estimates)
    alpha = 1 - confidence_level
    lower_quantile = alpha / 2
    upper_quantile = 1 - alpha / 2
    coefficients: list[ResampledCoefficient] = []

    for index, term in enumerate(terms):
        values = matrix[:, index]
        standard_error = float(values.std(ddof=1))
        lower = float(np.quantile(values, lower_quantile))
        upper = float(np.quantile(values, upper_quantile))

        coefficients.append(
            ResampledCoefficient(
                term=str(term),
                method="bootstrap",
                estimate=float(baseline.params[term]),
                standard_error=standard_error,
                confidence_interval_lower=lower,
                confidence_interval_upper=upper,
                p_value=None,
                successful_replications=len(estimates),
                requested_replications=replications,
            )
        )

    warnings: list[str] = []
    if failed:
        warnings.append(f"부트스트랩 반복 중 {failed}회가 실패했습니다.")

    return ResamplingReport(
        method="bootstrap",
        model_id=model_id,
        sample_size=len(model_data),
        coefficients=coefficients,
        warnings=warnings,
        metadata={
            "requested_replications": replications,
            "successful_replications": len(estimates),
            "failed_replications": failed,
            "confidence_level": confidence_level,
            "random_seed": random_seed,
        },
    )


def jackknife_ols(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    model_id: str = "main_model",
    confidence_level: float = 0.95,
) -> ResamplingReport:
    """Leave-one-out 잭나이프 OLS를 수행한다."""
    model_data, outcome, predictors = _prepare_ols_arrays(
        dataframe,
        dependent_variable,
        independent_variables,
    )

    if len(model_data) < 5:
        raise ValueError("잭나이프에는 최소 5개 사례가 필요합니다.")

    baseline = sm.OLS(outcome, predictors).fit()
    terms = list(baseline.params.index)
    leave_one_out_estimates: list[np.ndarray] = []

    for position in range(len(model_data)):
        keep = np.ones(len(model_data), dtype=bool)
        keep[position] = False

        fitted = sm.OLS(
            outcome.iloc[keep],
            predictors.iloc[keep],
        ).fit()
        leave_one_out_estimates.append(fitted.params.reindex(terms).to_numpy(dtype=float))

    matrix = np.vstack(leave_one_out_estimates)
    mean_estimate = matrix.mean(axis=0)
    sample_size = len(model_data)
    variance = (sample_size - 1) / sample_size * ((matrix - mean_estimate) ** 2).sum(axis=0)
    standard_errors = np.sqrt(variance)
    critical = 1.959963984540054
    coefficients: list[ResampledCoefficient] = []

    for index, term in enumerate(terms):
        estimate = float(baseline.params[term])
        standard_error = float(standard_errors[index])

        coefficients.append(
            ResampledCoefficient(
                term=str(term),
                method="jackknife",
                estimate=estimate,
                standard_error=standard_error,
                confidence_interval_lower=(estimate - critical * standard_error),
                confidence_interval_upper=(estimate + critical * standard_error),
                p_value=None,
                successful_replications=sample_size,
                requested_replications=sample_size,
            )
        )

    return ResamplingReport(
        method="jackknife",
        model_id=model_id,
        sample_size=sample_size,
        coefficients=coefficients,
        metadata={
            "replications": sample_size,
            "confidence_level": confidence_level,
        },
    )


def resampling_report_to_dataframe(
    report: ResamplingReport,
) -> pd.DataFrame:
    """재표집 결과를 데이터프레임으로 변환한다."""
    return pd.DataFrame([asdict(item) for item in report.coefficients])


def resampling_summary_to_dataframe(
    report: ResamplingReport,
) -> pd.DataFrame:
    """재표집 요약을 세로형 표로 변환한다."""
    values = {
        "method": report.method,
        "model_id": report.model_id,
        "sample_size": report.sample_size,
        "coefficient_count": len(report.coefficients),
        "warning_count": len(report.warnings),
        **report.metadata,
    }

    return pd.DataFrame(
        {
            "item": list(values.keys()),
            "value": list(values.values()),
        }
    )
