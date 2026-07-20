"""Random Intercept 혼합효과모형의 optimizer 민감도와 계수 안정성 평가."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from src.statistics.regression.mixed_effects import fit_random_intercept, fit_random_slope

DEFAULT_OPTIMIZERS = ("lbfgs", "bfgs", "cg", "powell")


@dataclass(slots=True)
class MixedCoefficientComparison:
    term: str
    optimizer: str
    estimate: float
    standard_error: float
    statistic: float
    p_value: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    significant: bool
    direction: str
    converged: bool


@dataclass(slots=True)
class MixedTermStability:
    term: str
    direction_consistent: bool
    significance_consistent: bool
    converged_model_count: int
    model_count: int
    estimate_minimum: float
    estimate_maximum: float
    standard_error_minimum: float
    standard_error_maximum: float
    status: str
    interpretation: str


@dataclass(slots=True)
class MixedEffectsRobustnessReport:
    model_id: str
    dependent_variable: str
    independent_variables: list[str]
    group_variable: str
    reml: bool
    optimizers: list[str]
    coefficient_comparisons: list[MixedCoefficientComparison]
    term_stability: list[MixedTermStability]
    model_statistics: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def build_mixed_effects_robustness_report(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    group_variable: str,
    model_id: str = "main_model",
    reml: bool = False,
    max_iterations: int = 200,
    alpha: float = 0.05,
    optimizers: tuple[str, ...] = DEFAULT_OPTIMIZERS,
    random_slope_variable: str | None = None,
) -> MixedEffectsRobustnessReport:
    """동일 Random Intercept 모형을 여러 optimizer로 반복 적합한다."""
    if not optimizers:
        raise ValueError("optimizer를 한 개 이상 지정해야 합니다.")

    results = []
    failed: list[str] = []
    for optimizer in optimizers:
        try:
            results.append(
                (
                    optimizer,
                    (fit_random_slope if random_slope_variable else fit_random_intercept)(
                        dataframe,
                        dependent_variable=dependent_variable,
                        independent_variables=independent_variables,
                        group_variable=group_variable,
                        model_id=f"{model_id}_{optimizer}",
                        reml=reml,
                        method=optimizer,
                        max_iterations=max_iterations,
                        **(
                            {"random_slope_variable": random_slope_variable}
                            if random_slope_variable
                            else {}
                        ),
                    ),
                )
            )
        except ValueError:
            failed.append(optimizer)

    if not results:
        raise ValueError("모든 optimizer에서 Random Intercept 모형 추정에 실패했습니다.")

    comparisons: list[MixedCoefficientComparison] = []
    for optimizer, result in results:
        for coefficient in result.coefficients:
            direction = (
                "positive"
                if coefficient.estimate > 0
                else "negative"
                if coefficient.estimate < 0
                else "zero"
            )
            comparisons.append(
                MixedCoefficientComparison(
                    term=coefficient.term,
                    optimizer=optimizer,
                    estimate=coefficient.estimate,
                    standard_error=coefficient.standard_error,
                    statistic=coefficient.statistic,
                    p_value=coefficient.p_value,
                    confidence_interval_lower=coefficient.confidence_interval_lower,
                    confidence_interval_upper=coefficient.confidence_interval_upper,
                    significant=coefficient.p_value < alpha,
                    direction=direction,
                    converged=result.converged,
                )
            )

    frame = pd.DataFrame([asdict(item) for item in comparisons])
    stability: list[MixedTermStability] = []
    warnings: list[str] = []

    for term, group in frame.groupby("term", sort=False):
        direction_consistent = group["direction"].nunique() == 1
        significance_consistent = group["significant"].nunique() == 1
        converged_model_count = int(group["converged"].sum())
        model_count = len(group)

        if (
            direction_consistent
            and significance_consistent
            and converged_model_count == model_count
        ):
            status = "STABLE"
            interpretation = "계수 방향과 유의성이 모든 수렴 optimizer에서 일관됩니다."
        elif direction_consistent:
            status = "PARTIALLY_STABLE"
            interpretation = (
                "계수 방향은 일관되지만 유의성 또는 수렴 결과가 optimizer에 따라 다릅니다."
            )
        else:
            status = "UNSTABLE"
            interpretation = "계수 방향이 optimizer에 따라 달라 해석에 주의가 필요합니다."

        stability.append(
            MixedTermStability(
                term=str(term),
                direction_consistent=direction_consistent,
                significance_consistent=significance_consistent,
                converged_model_count=converged_model_count,
                model_count=model_count,
                estimate_minimum=float(group["estimate"].min()),
                estimate_maximum=float(group["estimate"].max()),
                standard_error_minimum=float(group["standard_error"].min()),
                standard_error_maximum=float(group["standard_error"].max()),
                status=status,
                interpretation=interpretation,
            )
        )
        if status != "STABLE" and str(term).lower() not in {"const", "intercept"}:
            warnings.append(f"{term}: {interpretation}")

    model_statistics = pd.DataFrame(
        [
            {
                "optimizer": optimizer,
                "converged": result.converged,
                "sample_size": result.sample_size,
                "log_likelihood": result.fit_statistics.get("log_likelihood"),
                "aic": result.fit_statistics.get("aic"),
                "bic": result.fit_statistics.get("bic"),
                "random_intercept_variance": result.fit_statistics.get("random_intercept_variance"),
                "random_slope_variance": result.fit_statistics.get("random_slope_variance"),
                "random_intercept_slope_covariance": result.fit_statistics.get(
                    "random_intercept_slope_covariance"
                ),
                "random_intercept_slope_correlation": result.fit_statistics.get(
                    "random_intercept_slope_correlation"
                ),
                "residual_variance": result.fit_statistics.get("residual_variance"),
                "intraclass_correlation": result.fit_statistics.get("intraclass_correlation"),
                "warning_count": len(result.warnings),
            }
            for optimizer, result in results
        ]
    )

    if failed:
        warnings.append("추정에 실패한 optimizer: " + ", ".join(failed))

    substantive = [item for item in stability if item.term.lower() not in {"const", "intercept"}]
    stable_count = sum(item.status == "STABLE" for item in substantive)
    summary = {
        "model_id": model_id,
        "requested_optimizer_count": len(optimizers),
        "successful_optimizer_count": len(results),
        "failed_optimizer_count": len(failed),
        "converged_optimizer_count": int(model_statistics["converged"].sum()),
        "term_count": len(substantive),
        "stable_term_count": stable_count,
        "all_substantive_terms_stable": stable_count == len(substantive) if substantive else True,
        "group_variable": group_variable,
        "reml": reml,
        "random_slope_variable": random_slope_variable,
    }

    return MixedEffectsRobustnessReport(
        model_id=model_id,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        group_variable=group_variable,
        reml=reml,
        optimizers=list(optimizers),
        coefficient_comparisons=comparisons,
        term_stability=stability,
        model_statistics=model_statistics,
        warnings=warnings,
        summary=summary,
    )


def coefficient_comparison_to_dataframe(report: MixedEffectsRobustnessReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.coefficient_comparisons])


def stability_summary_to_dataframe(report: MixedEffectsRobustnessReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.term_stability])


def model_comparison_to_dataframe(report: MixedEffectsRobustnessReport) -> pd.DataFrame:
    return report.model_statistics.copy()


def robustness_summary_to_dataframe(report: MixedEffectsRobustnessReport) -> pd.DataFrame:
    values = {**report.summary, "warning_count": len(report.warnings)}
    return pd.DataFrame({"item": list(values), "value": list(values.values())})
