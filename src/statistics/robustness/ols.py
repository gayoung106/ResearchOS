"""OLS 표준오차 추정방식 비교 및 계수 안정성 평가."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from src.statistics.regression.ols import fit_ols

COVARIANCE_TYPES = ("nonrobust", "HC0", "HC1", "HC2", "HC3")


@dataclass(slots=True)
class CoefficientComparison:
    """동일 계수의 공분산 추정방식별 비교."""

    term: str
    covariance_type: str
    estimate: float
    standard_error: float
    statistic: float
    p_value: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    significant: bool
    direction: str


@dataclass(slots=True)
class TermStability:
    """계수별 강건성 평가."""

    term: str
    direction_consistent: bool
    significance_consistent: bool
    significant_model_count: int
    model_count: int
    estimate_minimum: float
    estimate_maximum: float
    standard_error_minimum: float
    standard_error_maximum: float
    status: str
    interpretation: str


@dataclass(slots=True)
class OLSRobustnessReport:
    """OLS 강건성 비교 전체 결과."""

    model_id: str
    dependent_variable: str
    independent_variables: list[str]
    covariance_types: list[str]
    coefficient_comparisons: list[CoefficientComparison]
    term_stability: list[TermStability]
    model_statistics: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def build_ols_robustness_report(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    model_id: str = "main_model",
    alpha: float = 0.05,
    covariance_types: tuple[str, ...] = COVARIANCE_TYPES,
) -> OLSRobustnessReport:
    """동일 OLS 모형을 여러 공분산 추정방식으로 반복 적합한다."""
    if not covariance_types:
        raise ValueError("공분산 추정방식을 한 개 이상 지정해야 합니다.")

    results = [
        fit_ols(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            model_id=f"{model_id}_{covariance_type.lower()}",
            covariance_type=covariance_type,
        )
        for covariance_type in covariance_types
    ]

    comparisons: list[CoefficientComparison] = []

    for covariance_type, result in zip(
        covariance_types,
        results,
        strict=True,
    ):
        for coefficient in result.coefficients:
            if coefficient.estimate > 0:
                direction = "positive"
            elif coefficient.estimate < 0:
                direction = "negative"
            else:
                direction = "zero"

            comparisons.append(
                CoefficientComparison(
                    term=coefficient.term,
                    covariance_type=covariance_type,
                    estimate=coefficient.estimate,
                    standard_error=coefficient.standard_error,
                    statistic=coefficient.statistic,
                    p_value=coefficient.p_value,
                    confidence_interval_lower=(coefficient.confidence_interval_lower),
                    confidence_interval_upper=(coefficient.confidence_interval_upper),
                    significant=coefficient.p_value < alpha,
                    direction=direction,
                )
            )

    comparison_frame = pd.DataFrame([asdict(item) for item in comparisons])

    term_stability: list[TermStability] = []
    warnings: list[str] = []

    for term, group in comparison_frame.groupby("term", sort=False):
        directions = set(group["direction"])
        significance_values = set(group["significant"].astype(bool))
        direction_consistent = len(directions) == 1
        significance_consistent = len(significance_values) == 1
        significant_model_count = int(group["significant"].sum())
        model_count = len(group)

        if direction_consistent and significance_consistent:
            status = "STABLE"
            interpretation = "계수 방향과 통계적 유의성이 모든 추정방식에서 일관됩니다."
        elif direction_consistent:
            status = "PARTIALLY_STABLE"
            interpretation = "계수 방향은 일관되지만 통계적 유의성은 추정방식에 따라 달라집니다."
        else:
            status = "UNSTABLE"
            interpretation = "계수 방향이 추정방식에 따라 달라져 해석에 주의가 필요합니다."

        term_stability.append(
            TermStability(
                term=str(term),
                direction_consistent=direction_consistent,
                significance_consistent=significance_consistent,
                significant_model_count=significant_model_count,
                model_count=model_count,
                estimate_minimum=float(group["estimate"].min()),
                estimate_maximum=float(group["estimate"].max()),
                standard_error_minimum=float(group["standard_error"].min()),
                standard_error_maximum=float(group["standard_error"].max()),
                status=status,
                interpretation=interpretation,
            )
        )

        if status != "STABLE" and str(term).lower() not in {
            "const",
            "intercept",
        }:
            warnings.append(f"{term}: {interpretation}")

    model_statistics = pd.DataFrame(
        [
            {
                "covariance_type": covariance_type,
                "sample_size": result.sample_size,
                "r_squared": result.fit_statistics.get("r_squared"),
                "adjusted_r_squared": result.fit_statistics.get("adjusted_r_squared"),
                "aic": result.fit_statistics.get("aic"),
                "bic": result.fit_statistics.get("bic"),
            }
            for covariance_type, result in zip(
                covariance_types,
                results,
                strict=True,
            )
        ]
    )

    substantive_terms = [
        item for item in term_stability if item.term.lower() not in {"const", "intercept"}
    ]

    stable_count = sum(item.status == "STABLE" for item in substantive_terms)

    summary = {
        "model_id": model_id,
        "covariance_type_count": len(covariance_types),
        "term_count": len(substantive_terms),
        "stable_term_count": stable_count,
        "partially_stable_term_count": sum(
            item.status == "PARTIALLY_STABLE" for item in substantive_terms
        ),
        "unstable_term_count": sum(item.status == "UNSTABLE" for item in substantive_terms),
        "all_substantive_terms_stable": (
            stable_count == len(substantive_terms) if substantive_terms else True
        ),
    }

    return OLSRobustnessReport(
        model_id=model_id,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        covariance_types=list(covariance_types),
        coefficient_comparisons=comparisons,
        term_stability=term_stability,
        model_statistics=model_statistics,
        warnings=warnings,
        summary=summary,
    )


def coefficient_comparison_to_dataframe(
    report: OLSRobustnessReport,
) -> pd.DataFrame:
    """계수 비교표를 반환한다."""
    return pd.DataFrame([asdict(item) for item in report.coefficient_comparisons])


def stability_summary_to_dataframe(
    report: OLSRobustnessReport,
) -> pd.DataFrame:
    """계수별 안정성 평가표를 반환한다."""
    return pd.DataFrame([asdict(item) for item in report.term_stability])


def model_comparison_to_dataframe(
    report: OLSRobustnessReport,
) -> pd.DataFrame:
    """모형 수준 비교표를 반환한다."""
    return report.model_statistics.copy()


def robustness_summary_to_dataframe(
    report: OLSRobustnessReport,
) -> pd.DataFrame:
    """강건성 요약을 세로형 표로 반환한다."""
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
