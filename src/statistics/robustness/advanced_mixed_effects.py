"""Random Intercept 모형의 집단 단위 재표집 강건성 분석."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.statistics.regression.mixed_effects import fit_random_intercept, fit_random_slope


@dataclass(slots=True)
class MixedResampledCoefficient:
    method: str
    term: str
    estimate: float
    standard_error: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    p_value: float | None
    successful_replications: int
    requested_replications: int
    direction_consistency: float
    significance_consistency: float


@dataclass(slots=True)
class MixedAdvancedRobustnessReport:
    model_id: str
    dependent_variable: str
    independent_variables: list[str]
    group_variable: str
    group_count: int
    coefficients: list[MixedResampledCoefficient]
    leave_one_group_out: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _complete_data(
    dataframe: pd.DataFrame,
    dependent_variable: str,
    independent_variables: list[str],
    group_variable: str,
) -> pd.DataFrame:
    columns = [dependent_variable, *independent_variables, group_variable]
    frame = dataframe[columns].copy()
    for column in [dependent_variable, *independent_variables]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna()
    if frame[group_variable].nunique() < 3:
        raise ValueError("고급 Mixed 강건성 분석에는 최소 3개 그룹이 필요합니다.")
    return frame


def build_mixed_advanced_robustness_report(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    group_variable: str,
    model_id: str = "main_model",
    bootstrap_replications: int = 500,
    run_leave_one_group_out: bool = True,
    confidence_level: float = 0.95,
    random_seed: int = 2026,
    reml: bool = False,
    optimizer: str = "lbfgs",
    max_iterations: int = 200,
    random_slope_variable: str | None = None,
) -> MixedAdvancedRobustnessReport:
    """그룹 부트스트랩과 Leave-One-Group-Out 분석을 수행한다."""
    if bootstrap_replications < 20:
        raise ValueError("Mixed 그룹 부트스트랩 반복 수는 최소 20회여야 합니다.")
    frame = _complete_data(dataframe, dependent_variable, independent_variables, group_variable)
    groups = list(frame[group_variable].drop_duplicates())
    baseline = (fit_random_slope if random_slope_variable else fit_random_intercept)(
        frame,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        group_variable=group_variable,
        model_id=model_id,
        reml=reml,
        method=optimizer,
        max_iterations=max_iterations,
    )
    baseline_by_term = {item.term: item for item in baseline.coefficients}
    rng = np.random.default_rng(random_seed)
    draws: dict[str, list[tuple[float, float]]] = {term: [] for term in baseline_by_term}
    failures = 0

    for replication in range(bootstrap_replications):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        pieces = []
        for position, selected_group in enumerate(sampled_groups):
            piece = frame.loc[frame[group_variable] == selected_group].copy()
            piece[group_variable] = f"{position}_{selected_group}"
            pieces.append(piece)
        sample = pd.concat(pieces, ignore_index=True)
        try:
            result = (fit_random_slope if random_slope_variable else fit_random_intercept)(
                sample,
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                group_variable=group_variable,
                model_id=f"{model_id}_group_bootstrap_{replication}",
                reml=reml,
                method=optimizer,
                max_iterations=max_iterations,
                **(
                    {"random_slope_variable": random_slope_variable}
                    if random_slope_variable
                    else {}
                ),
            )
        except (ValueError, np.linalg.LinAlgError):
            failures += 1
            continue
        for coefficient in result.coefficients:
            if coefficient.term in draws:
                draws[coefficient.term].append((coefficient.estimate, coefficient.p_value))

    alpha = 1.0 - confidence_level
    coefficients: list[MixedResampledCoefficient] = []
    for term, values in draws.items():
        if not values:
            continue
        estimates = np.asarray([value[0] for value in values], dtype=float)
        p_values = np.asarray([value[1] for value in values], dtype=float)
        baseline_coefficient = baseline_by_term[term]
        baseline_direction = np.sign(baseline_coefficient.estimate)
        baseline_significant = baseline_coefficient.p_value < alpha
        coefficients.append(
            MixedResampledCoefficient(
                method="group_bootstrap",
                term=term,
                estimate=float(estimates.mean()),
                standard_error=float(estimates.std(ddof=1)) if len(estimates) > 1 else 0.0,
                confidence_interval_lower=float(np.quantile(estimates, alpha / 2)),
                confidence_interval_upper=float(np.quantile(estimates, 1 - alpha / 2)),
                p_value=None,
                successful_replications=len(estimates),
                requested_replications=bootstrap_replications,
                direction_consistency=float(np.mean(np.sign(estimates) == baseline_direction)),
                significance_consistency=float(np.mean((p_values < alpha) == baseline_significant)),
            )
        )

    logo_rows: list[dict[str, Any]] = []
    logo_failures = 0
    if run_leave_one_group_out:
        for omitted_group in groups:
            reduced = frame.loc[frame[group_variable] != omitted_group]
            try:
                result = (fit_random_slope if random_slope_variable else fit_random_intercept)(
                    reduced,
                    dependent_variable=dependent_variable,
                    independent_variables=independent_variables,
                    group_variable=group_variable,
                    model_id=f"{model_id}_without_{omitted_group}",
                    reml=reml,
                    method=optimizer,
                    max_iterations=max_iterations,
                )
            except (ValueError, np.linalg.LinAlgError):
                logo_failures += 1
                continue
            for coefficient in result.coefficients:
                logo_rows.append(
                    {
                        "omitted_group": omitted_group,
                        "term": coefficient.term,
                        "estimate": coefficient.estimate,
                        "standard_error": coefficient.standard_error,
                        "p_value": coefficient.p_value,
                        "converged": result.converged,
                        "random_intercept_variance": result.fit_statistics.get(
                            "random_intercept_variance"
                        ),
                        "residual_variance": result.fit_statistics.get("residual_variance"),
                        "intraclass_correlation": result.fit_statistics.get(
                            "intraclass_correlation"
                        ),
                    }
                )

    warnings: list[str] = []
    success_rate = (bootstrap_replications - failures) / bootstrap_replications
    if success_rate < 0.8:
        warnings.append(f"그룹 부트스트랩 성공률이 {success_rate:.1%}로 낮습니다.")
    if logo_failures:
        warnings.append(f"Leave-One-Group-Out 추정 실패 그룹 수: {logo_failures}")
    for item in coefficients:
        if item.term.lower() not in {"const", "intercept"} and item.direction_consistency < 0.9:
            warnings.append(f"{item.term}: 그룹 부트스트랩에서 계수 방향 일관성이 낮습니다.")

    return MixedAdvancedRobustnessReport(
        model_id=model_id,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        group_variable=group_variable,
        group_count=len(groups),
        coefficients=coefficients,
        leave_one_group_out=pd.DataFrame(logo_rows),
        warnings=warnings,
        metadata={
            "bootstrap_replications": bootstrap_replications,
            "successful_bootstrap_replications": bootstrap_replications - failures,
            "bootstrap_success_rate": success_rate,
            "run_leave_one_group_out": run_leave_one_group_out,
            "successful_leave_one_group_out": len(groups) - logo_failures
            if run_leave_one_group_out
            else 0,
            "reml": reml,
            "optimizer": optimizer,
        },
    )


def mixed_resampling_to_dataframe(report: MixedAdvancedRobustnessReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.coefficients])


def mixed_advanced_summary_to_dataframe(report: MixedAdvancedRobustnessReport) -> pd.DataFrame:
    values = {
        "model_id": report.model_id,
        "group_variable": report.group_variable,
        "group_count": report.group_count,
        "coefficient_count": len(report.coefficients),
        "warning_count": len(report.warnings),
        **report.metadata,
    }
    return pd.DataFrame({"item": list(values), "value": list(values.values())})
