"""혼합효과 회귀모형 진단."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.statistics.diagnostics.ols import DiagnosticTestResult
from src.statistics.regression.base import RegressionResult


@dataclass(slots=True)
class MixedEffectsDiagnosticsReport:
    """혼합효과 모형의 진단 결과."""

    model_id: str
    sample_size: int
    group_count: int
    parameter_count: int
    diagnostic_tests: list[DiagnosticTestResult]
    residuals: pd.DataFrame
    group_residuals: pd.DataFrame
    random_effects: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _validate_mixed_effects_result(result: RegressionResult) -> Any:
    """지원되는 Mixed Effects 결과인지 확인한다."""
    if result.model_type not in {
        "mixed_random_intercept",
        "mixed_random_slope",
        "mixed_three_level",
    }:
        raise ValueError(
            "Mixed Effects 진단은 model_type='mixed_random_intercept' 또는 'mixed_random_slope' 결과에만 적용할 수 있습니다."
        )
    if result.raw_result is None:
        raise ValueError("원본 statsmodels 결과 객체가 없습니다.")
    return result.raw_result


def _status_from_p_value(p_value: float | None, *, alpha: float) -> str:
    if p_value is None or not np.isfinite(p_value):
        return "UNAVAILABLE"
    return "WARNING" if p_value < alpha else "PASS"


def _row_labels(fitted: Any, sample_size: int) -> list[Any]:
    labels = getattr(getattr(fitted.model, "data", None), "row_labels", None)
    if labels is None or len(labels) != sample_size:
        return list(range(sample_size))
    return list(labels)


def calculate_mixed_effects_residuals(
    result: RegressionResult,
    *,
    standardized_residual_threshold: float = 3.0,
) -> pd.DataFrame:
    """조건부 적합값과 잔차를 사례 단위로 정리한다."""
    fitted = _validate_mixed_effects_result(result)
    residuals = np.asarray(fitted.resid, dtype=float)
    fitted_values = np.asarray(fitted.fittedvalues, dtype=float)
    groups = np.asarray(fitted.model.groups)
    residual_scale = float(np.sqrt(max(float(fitted.scale), np.finfo(float).eps)))
    standardized = residuals / residual_scale

    output = pd.DataFrame(
        {
            "row_index": _row_labels(fitted, len(residuals)),
            "group": groups,
            "fitted_value": fitted_values,
            "residual": residuals,
            "standardized_residual": standardized,
        }
    )
    output["large_standardized_residual"] = (
        output["standardized_residual"].abs() > standardized_residual_threshold
    )
    return output


def calculate_group_residual_summary(
    result: RegressionResult,
    *,
    group_standardized_mean_threshold: float = 2.0,
) -> pd.DataFrame:
    """그룹별 잔차 수준과 체계적 편향 가능성을 요약한다."""
    residuals = calculate_mixed_effects_residuals(result)
    residual_standard_deviation = float(residuals["residual"].std(ddof=1))

    grouped = (
        residuals.groupby("group", dropna=False, sort=False)
        .agg(
            group_size=("residual", "size"),
            mean_residual=("residual", "mean"),
            residual_standard_deviation=("residual", "std"),
            root_mean_squared_residual=(
                "residual",
                lambda values: float(np.sqrt(np.mean(np.square(values)))),
            ),
            maximum_absolute_standardized_residual=(
                "standardized_residual",
                lambda values: float(np.max(np.abs(values))),
            ),
            large_residual_count=("large_standardized_residual", "sum"),
        )
        .reset_index()
    )

    denominator = residual_standard_deviation / np.sqrt(grouped["group_size"])
    denominator = denominator.where(
        denominator > np.finfo(float).eps,
        np.nan,
    )
    grouped["standardized_mean_residual"] = grouped["mean_residual"] / denominator
    grouped["group_residual_flag"] = (
        grouped["standardized_mean_residual"].abs() > group_standardized_mean_threshold
    )
    grouped["large_residual_count"] = grouped["large_residual_count"].astype(int)
    return grouped


def calculate_random_effects(result: RegressionResult) -> pd.DataFrame:
    """그룹별 Random Intercept 및 Random Slope 추정치를 정리한다."""
    fitted = _validate_mixed_effects_result(result)
    slope_terms = list(result.metadata.get("random_slope_variables") or [])
    if not slope_terms and result.metadata.get("random_slope_variable"):
        slope_terms = [str(result.metadata["random_slope_variable"])]
    rows: list[dict[str, Any]] = []
    for group, effect in fitted.random_effects.items():
        values = np.asarray(effect, dtype=float).reshape(-1)
        if values.size == 0:
            continue
        row: dict[str, Any] = {"group": group, "random_intercept": float(values[0])}
        for index, term in enumerate(slope_terms, start=1):
            if values.size > index:
                row[f"random_slope__{term}"] = float(values[index])
        if len(slope_terms) == 1 and f"random_slope__{slope_terms[0]}" in row:
            row["random_slope"] = row[f"random_slope__{slope_terms[0]}"]
        rows.append(row)
    columns = ["group", "random_intercept", *[f"random_slope__{v}" for v in slope_terms]]
    if len(slope_terms) == 1:
        columns.append("random_slope")
    output = pd.DataFrame(rows, columns=columns)
    if not output.empty:
        output["absolute_random_intercept"] = output["random_intercept"].abs()
        for term in slope_terms:
            column = f"random_slope__{term}"
            if column in output:
                output[f"absolute_{column}"] = output[column].abs()
        output = output.sort_values("absolute_random_intercept", ascending=False, ignore_index=True)
    else:
        output["absolute_random_intercept"] = pd.Series(dtype=float)
    return output


def run_mixed_effects_diagnostic_tests(
    result: RegressionResult,
    *,
    alpha: float = 0.05,
) -> list[DiagnosticTestResult]:
    """조건부 잔차와 Random Intercept의 정규성을 검사한다."""
    residuals = calculate_mixed_effects_residuals(result)["residual"].to_numpy()
    random_effects = calculate_random_effects(result)["random_intercept"].to_numpy()
    tests: list[DiagnosticTestResult] = []

    residual_statistic, residual_p_value = stats.jarque_bera(residuals)
    tests.append(
        DiagnosticTestResult(
            test_name="Conditional Residual Jarque-Bera",
            statistic=float(residual_statistic),
            p_value=float(residual_p_value),
            status=_status_from_p_value(float(residual_p_value), alpha=alpha),
            interpretation=("유의하면 조건부 잔차의 정규성 가정에서 벗어날 가능성이 있습니다."),
        )
    )

    if len(random_effects) >= 8 and not np.isclose(np.var(random_effects), 0.0):
        effect_statistic, effect_p_value = stats.jarque_bera(random_effects)
        tests.append(
            DiagnosticTestResult(
                test_name="Random Intercept Jarque-Bera",
                statistic=float(effect_statistic),
                p_value=float(effect_p_value),
                status=_status_from_p_value(float(effect_p_value), alpha=alpha),
                interpretation=(
                    "유의하면 Random Intercept 정규성 가정에서 벗어날 가능성이 있습니다."
                ),
            )
        )
    else:
        tests.append(
            DiagnosticTestResult(
                test_name="Random Intercept Jarque-Bera",
                statistic=None,
                p_value=None,
                status="UNAVAILABLE",
                interpretation=(
                    "그룹 수가 8개 미만이거나 Random Intercept 분산이 없어 "
                    "정규성 검정을 계산하지 않았습니다."
                ),
            )
        )

    if result.model_type == "mixed_random_slope":
        random_effect_table = calculate_random_effects(result)
        slope_columns = [c for c in random_effect_table.columns if c.startswith("random_slope__")]
        for column in slope_columns:
            term = column.removeprefix("random_slope__")
            slope_effects = random_effect_table[column].dropna().to_numpy()
            if len(slope_effects) >= 8 and not np.isclose(np.var(slope_effects), 0.0):
                statistic, p_value = stats.jarque_bera(slope_effects)
                tests.append(
                    DiagnosticTestResult(
                        test_name=f"Random Slope ({term}) Jarque-Bera",
                        statistic=float(statistic),
                        p_value=float(p_value),
                        status=_status_from_p_value(float(p_value), alpha=alpha),
                        interpretation=f"유의하면 {term} Random Slope 정규성 가정에서 벗어날 가능성이 있습니다.",
                    )
                )
            else:
                tests.append(
                    DiagnosticTestResult(
                        test_name=f"Random Slope ({term}) Jarque-Bera",
                        statistic=None,
                        p_value=None,
                        status="UNAVAILABLE",
                        interpretation=f"그룹 수가 8개 미만이거나 {term} Random Slope 분산이 없어 정규성 검정을 계산하지 않았습니다.",
                    )
                )

    return tests


def build_mixed_effects_diagnostics(
    result: RegressionResult,
    *,
    alpha: float = 0.05,
    standardized_residual_threshold: float = 3.0,
    group_standardized_mean_threshold: float = 2.0,
    singular_variance_tolerance: float = 1e-8,
) -> MixedEffectsDiagnosticsReport:
    """혼합효과 모형의 종합 진단 보고서를 만든다."""
    fitted = _validate_mixed_effects_result(result)
    residuals = calculate_mixed_effects_residuals(
        result,
        standardized_residual_threshold=standardized_residual_threshold,
    )
    group_residuals = calculate_group_residual_summary(
        result,
        group_standardized_mean_threshold=group_standardized_mean_threshold,
    )
    random_effects = calculate_random_effects(result)
    diagnostic_tests = run_mixed_effects_diagnostic_tests(result, alpha=alpha)

    random_intercept_variance = float(
        result.fit_statistics.get(
            "random_intercept_variance",
            fitted.cov_re.iloc[0, 0],
        )
    )
    random_slope_variance = result.fit_statistics.get("random_slope_variance")
    random_slope_variances = result.fit_statistics.get("random_slope_variances") or (
        {result.metadata.get("random_slope_variable"): random_slope_variance}
        if random_slope_variance is not None
        else {}
    )
    covariance_min_eigenvalue = result.fit_statistics.get("random_effect_covariance_min_eigenvalue")
    near_zero_slope_terms = [
        str(term)
        for term, value in random_slope_variances.items()
        if value is not None and float(value) <= singular_variance_tolerance
    ]
    near_zero_slope_variance = bool(near_zero_slope_terms)
    singular_fit = (
        random_intercept_variance <= singular_variance_tolerance
        or near_zero_slope_variance
        or (
            covariance_min_eigenvalue is not None
            and float(covariance_min_eigenvalue) <= singular_variance_tolerance
        )
    )
    large_residual_count = int(residuals["large_standardized_residual"].sum())
    biased_group_count = int(group_residuals["group_residual_flag"].sum())
    warnings: list[str] = []

    for test in diagnostic_tests:
        if test.status == "WARNING":
            warnings.append(f"{test.test_name}: {test.interpretation}")

    if not bool(result.converged):
        warnings.append("혼합효과 모형이 수렴하지 않았습니다.")
    if singular_fit:
        warnings.append("랜덤효과 공분산 구조가 특이하거나 거의 특이할 가능성이 있습니다.")
    if near_zero_slope_variance:
        warnings.append("Random Slope 분산이 0에 가까워 기울기 이질성이 거의 없을 수 있습니다.")
    if large_residual_count:
        warnings.append(
            "표준화 잔차 절대값이 "
            f"{standardized_residual_threshold:g}을 초과한 사례가 "
            f"{large_residual_count}개 있습니다."
        )
    if biased_group_count:
        warnings.append(
            "표준화 평균 잔차 절대값이 "
            f"{group_standardized_mean_threshold:g}을 초과한 그룹이 "
            f"{biased_group_count}개 있습니다."
        )

    summary = {
        "model_id": result.model_id,
        "sample_size": int(fitted.nobs),
        "group_count": int(len(random_effects)),
        "parameter_count": int(len(fitted.params)),
        "converged": bool(result.converged),
        "singular_fit": bool(singular_fit),
        "random_intercept_variance": random_intercept_variance,
        "random_slope_variance": random_slope_variance,
        "random_slope_variances": random_slope_variances,
        "near_zero_slope_terms": near_zero_slope_terms,
        "random_intercept_slope_covariance": result.fit_statistics.get(
            "random_intercept_slope_covariance"
        ),
        "random_intercept_slope_correlation": result.fit_statistics.get(
            "random_intercept_slope_correlation"
        ),
        "near_zero_slope_variance": bool(near_zero_slope_variance),
        "residual_variance": float(fitted.scale),
        "intraclass_correlation": result.fit_statistics.get("intraclass_correlation"),
        "large_residual_count": large_residual_count,
        "biased_group_count": biased_group_count,
        "diagnostic_warning_count": sum(item.status == "WARNING" for item in diagnostic_tests),
    }

    return MixedEffectsDiagnosticsReport(
        model_id=result.model_id,
        sample_size=int(fitted.nobs),
        group_count=int(len(random_effects)),
        parameter_count=int(len(fitted.params)),
        diagnostic_tests=diagnostic_tests,
        residuals=residuals,
        group_residuals=group_residuals,
        random_effects=random_effects,
        warnings=warnings,
        summary=summary,
    )


def mixed_effects_tests_to_dataframe(
    report: MixedEffectsDiagnosticsReport,
) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.diagnostic_tests])


def mixed_effects_residuals_to_dataframe(
    report: MixedEffectsDiagnosticsReport,
) -> pd.DataFrame:
    return report.residuals.copy()


def mixed_effects_group_residuals_to_dataframe(
    report: MixedEffectsDiagnosticsReport,
) -> pd.DataFrame:
    return report.group_residuals.copy()


def mixed_effects_random_effects_to_dataframe(
    report: MixedEffectsDiagnosticsReport,
) -> pd.DataFrame:
    return report.random_effects.copy()


def mixed_effects_diagnostic_summary_to_dataframe(
    report: MixedEffectsDiagnosticsReport,
) -> pd.DataFrame:
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
