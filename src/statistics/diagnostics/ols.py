"""OLS 회귀모형의 다중공선성·잔차·이분산성·영향력 진단."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.statistics.regression.base import RegressionResult


@dataclass(slots=True)
class DiagnosticTestResult:
    test_name: str
    statistic: float | None
    p_value: float | None
    status: str
    interpretation: str


@dataclass(slots=True)
class MulticollinearityResult:
    variable_name: str
    vif: float | None
    tolerance: float | None
    status: str
    interpretation: str


@dataclass(slots=True)
class InfluenceThresholds:
    cooks_distance: float
    leverage: float
    dfbetas: float
    dffits: float
    studentized_residual: float = 3.0


@dataclass(slots=True)
class OLSDiagnosticsReport:
    model_id: str
    sample_size: int
    parameter_count: int
    multicollinearity: list[MulticollinearityResult]
    diagnostic_tests: list[DiagnosticTestResult]
    residuals: pd.DataFrame
    influence: pd.DataFrame
    thresholds: InfluenceThresholds
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _validate_ols_result(result: RegressionResult) -> Any:
    if result.model_type != "ols":
        raise ValueError("OLS 진단은 model_type='ols' 결과에만 적용할 수 있습니다.")
    if result.raw_result is None:
        raise ValueError("원본 statsmodels 결과 객체가 없습니다.")
    return result.raw_result


def _status_from_p_value(p_value: float | None, *, alpha: float) -> str:
    if p_value is None or np.isnan(p_value):
        return "UNAVAILABLE"
    return "WARNING" if p_value < alpha else "PASS"


def calculate_multicollinearity(
    result: RegressionResult,
) -> list[MulticollinearityResult]:
    fitted = _validate_ols_result(result)
    exog = np.asarray(fitted.model.exog, dtype=float)
    exog_names = list(fitted.model.exog_names)
    output: list[MulticollinearityResult] = []

    for index, variable_name in enumerate(exog_names):
        if variable_name.lower() in {"const", "intercept"}:
            continue

        try:
            vif = float(variance_inflation_factor(exog, index))
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            vif = np.inf

        tolerance = 0.0 if not np.isfinite(vif) or np.isclose(vif, 0.0) else 1.0 / vif

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


def run_diagnostic_tests(
    result: RegressionResult,
    *,
    alpha: float = 0.05,
) -> list[DiagnosticTestResult]:
    fitted = _validate_ols_result(result)
    residuals = np.asarray(fitted.resid, dtype=float)
    exog = np.asarray(fitted.model.exog, dtype=float)
    tests: list[DiagnosticTestResult] = []

    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(residuals, exog)
    tests.extend(
        [
            DiagnosticTestResult(
                "Breusch-Pagan LM",
                float(bp_lm),
                float(bp_lm_p),
                _status_from_p_value(float(bp_lm_p), alpha=alpha),
                "유의하면 이분산성 가능성이 있습니다.",
            ),
            DiagnosticTestResult(
                "Breusch-Pagan F",
                float(bp_f),
                float(bp_f_p),
                _status_from_p_value(float(bp_f_p), alpha=alpha),
                "유의하면 이분산성 가능성이 있습니다.",
            ),
        ]
    )

    try:
        white_lm, white_lm_p, white_f, white_f_p = het_white(residuals, exog)
        tests.extend(
            [
                DiagnosticTestResult(
                    "White LM",
                    float(white_lm),
                    float(white_lm_p),
                    _status_from_p_value(float(white_lm_p), alpha=alpha),
                    "유의하면 일반적 형태의 이분산성이 의심됩니다.",
                ),
                DiagnosticTestResult(
                    "White F",
                    float(white_f),
                    float(white_f_p),
                    _status_from_p_value(float(white_f_p), alpha=alpha),
                    "유의하면 일반적 형태의 이분산성이 의심됩니다.",
                ),
            ]
        )
    except (AssertionError, ValueError, np.linalg.LinAlgError):
        tests.append(
            DiagnosticTestResult(
                "White LM",
                None,
                None,
                "UNAVAILABLE",
                "White 검정을 계산할 수 없습니다.",
            )
        )

    try:
        reset_result = linear_reset(fitted, power=2, use_f=True)
        reset_statistic = float(np.asarray(reset_result.fvalue).squeeze())
        reset_p_value = float(np.asarray(reset_result.pvalue).squeeze())
        tests.append(
            DiagnosticTestResult(
                "Ramsey RESET",
                reset_statistic,
                reset_p_value,
                _status_from_p_value(reset_p_value, alpha=alpha),
                "유의하면 함수형태 오류 또는 누락변수 가능성이 있습니다.",
            )
        )
    except (ValueError, np.linalg.LinAlgError):
        tests.append(
            DiagnosticTestResult(
                "Ramsey RESET",
                None,
                None,
                "UNAVAILABLE",
                "RESET 검정을 계산할 수 없습니다.",
            )
        )

    jb_statistic, jb_p_value = stats.jarque_bera(residuals)
    tests.append(
        DiagnosticTestResult(
            "Jarque-Bera",
            float(jb_statistic),
            float(jb_p_value),
            _status_from_p_value(float(jb_p_value), alpha=alpha),
            "유의하면 잔차 정규성 가정에서 벗어날 가능성이 있습니다.",
        )
    )

    return tests


def calculate_residuals_and_influence(
    result: RegressionResult,
) -> tuple[pd.DataFrame, pd.DataFrame, InfluenceThresholds]:
    fitted = _validate_ols_result(result)
    influence = fitted.get_influence()

    sample_size = int(fitted.nobs)
    parameter_count = int(len(fitted.params))
    thresholds = InfluenceThresholds(
        cooks_distance=4 / sample_size,
        leverage=2 * parameter_count / sample_size,
        dfbetas=2 / np.sqrt(sample_size),
        dffits=2 * np.sqrt(parameter_count / sample_size),
    )

    residuals = pd.DataFrame(
        {
            "row_index": fitted.model.data.row_labels,
            "fitted_value": np.asarray(fitted.fittedvalues),
            "residual": np.asarray(fitted.resid),
            "standardized_residual": np.asarray(influence.resid_studentized_internal),
            "studentized_residual": np.asarray(influence.resid_studentized_external),
        }
    )
    residuals["large_studentized_residual"] = (
        residuals["studentized_residual"].abs() > thresholds.studentized_residual
    )

    cooks_distance = np.asarray(influence.cooks_distance[0], dtype=float)
    leverage = np.asarray(influence.hat_matrix_diag, dtype=float)
    dffits = np.asarray(influence.dffits[0], dtype=float)
    dfbetas = np.asarray(influence.dfbetas, dtype=float)

    influence_table = pd.DataFrame(
        {
            "row_index": fitted.model.data.row_labels,
            "cooks_distance": cooks_distance,
            "leverage": leverage,
            "dffits": dffits,
            "max_absolute_dfbetas": np.max(np.abs(dfbetas), axis=1),
        }
    )
    influence_table["cooks_flag"] = influence_table["cooks_distance"] > thresholds.cooks_distance
    influence_table["leverage_flag"] = influence_table["leverage"] > thresholds.leverage
    influence_table["dffits_flag"] = influence_table["dffits"].abs() > thresholds.dffits
    influence_table["dfbetas_flag"] = influence_table["max_absolute_dfbetas"] > thresholds.dfbetas
    influence_table["any_influence_flag"] = influence_table[
        ["cooks_flag", "leverage_flag", "dffits_flag", "dfbetas_flag"]
    ].any(axis=1)

    return residuals, influence_table, thresholds


def build_ols_diagnostics(
    result: RegressionResult,
    *,
    alpha: float = 0.05,
) -> OLSDiagnosticsReport:
    fitted = _validate_ols_result(result)
    multicollinearity = calculate_multicollinearity(result)
    diagnostic_tests = run_diagnostic_tests(result, alpha=alpha)
    residuals, influence, thresholds = calculate_residuals_and_influence(result)
    warnings: list[str] = []

    for item in multicollinearity:
        if item.status in {"WARNING", "FAIL"}:
            warnings.append(f"{item.variable_name}: {item.interpretation}")

    for test in diagnostic_tests:
        if test.status == "WARNING":
            warnings.append(f"{test.test_name}: {test.interpretation}")

    influence_count = int(influence["any_influence_flag"].sum())
    large_residual_count = int(residuals["large_studentized_residual"].sum())

    if influence_count:
        warnings.append(f"영향력 기준을 하나 이상 초과한 사례가 {influence_count}개 있습니다.")
    if large_residual_count:
        warnings.append(
            f"외적 학생화 잔차 절대값이 3을 초과한 사례가 {large_residual_count}개 있습니다."
        )

    summary = {
        "model_id": result.model_id,
        "sample_size": int(fitted.nobs),
        "parameter_count": int(len(fitted.params)),
        "vif_warning_count": sum(item.status in {"WARNING", "FAIL"} for item in multicollinearity),
        "diagnostic_warning_count": sum(item.status == "WARNING" for item in diagnostic_tests),
        "influential_case_count": influence_count,
        "large_residual_count": large_residual_count,
    }

    return OLSDiagnosticsReport(
        model_id=result.model_id,
        sample_size=int(fitted.nobs),
        parameter_count=int(len(fitted.params)),
        multicollinearity=multicollinearity,
        diagnostic_tests=diagnostic_tests,
        residuals=residuals,
        influence=influence,
        thresholds=thresholds,
        warnings=warnings,
        summary=summary,
    )


def multicollinearity_to_dataframe(
    report: OLSDiagnosticsReport,
) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.multicollinearity])


def tests_to_dataframe(report: OLSDiagnosticsReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.diagnostic_tests])


def residuals_to_dataframe(report: OLSDiagnosticsReport) -> pd.DataFrame:
    return report.residuals.copy()


def influence_to_dataframe(report: OLSDiagnosticsReport) -> pd.DataFrame:
    return report.influence.copy()


def diagnostic_summary_to_dataframe(
    report: OLSDiagnosticsReport,
) -> pd.DataFrame:
    values = {
        **report.summary,
        "cooks_distance_threshold": report.thresholds.cooks_distance,
        "leverage_threshold": report.thresholds.leverage,
        "dfbetas_threshold": report.thresholds.dfbetas,
        "dffits_threshold": report.thresholds.dffits,
        "studentized_residual_threshold": (report.thresholds.studentized_residual),
        "warning_count": len(report.warnings),
    }
    return pd.DataFrame({"item": list(values.keys()), "value": list(values.values())})
