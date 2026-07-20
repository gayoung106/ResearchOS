"""Robustness checks for non-Gaussian GLMM regression results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.statistics.regression.base import RegressionResult
from src.statistics.regression.selector import fit_regression_by_level

GLMM_MODEL_TYPES = {
    "mixed_binary_logit_random_intercept",
    "mixed_binary_logit_random_slope",
    "mixed_binary_logit_three_level",
    "mixed_poisson_random_intercept",
    "mixed_poisson_random_slope",
    "mixed_poisson_three_level",
    "mixed_negative_binomial_random_intercept",
    "mixed_negative_binomial_random_slope",
    "mixed_negative_binomial_three_level",
}

THREE_LEVEL_GLMM_MODEL_TYPES = {
    "mixed_binary_logit_three_level",
    "mixed_poisson_three_level",
    "mixed_negative_binomial_three_level",
}

NEGATIVE_BINOMIAL_GLMM_MODEL_TYPES = {
    "mixed_negative_binomial_random_intercept",
    "mixed_negative_binomial_random_slope",
    "mixed_negative_binomial_three_level",
}


@dataclass(slots=True)
class GLMMCoefficientComparison:
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
class GLMMTermStability:
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
class GLMMRobustnessReport:
    model_id: str
    model_type: str
    dependent_variable: str
    independent_variables: list[str]
    optimizers: list[str]
    coefficient_comparisons: list[GLMMCoefficientComparison]
    term_stability: list[GLMMTermStability]
    model_statistics: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _measurement_level(model_type: str) -> str:
    if model_type in {
        "mixed_binary_logit_random_intercept",
        "mixed_binary_logit_random_slope",
        "mixed_binary_logit_three_level",
    }:
        return "binary"
    return "count"


def _fit_options_from_result(result: RegressionResult, optimizer: str) -> dict[str, object]:
    options: dict[str, object] = {
        "optimizer": optimizer,
        "add_intercept": result.metadata.get("add_intercept", True),
        "max_iterations": result.metadata.get("max_iterations", 200),
    }
    if result.model_type in THREE_LEVEL_GLMM_MODEL_TYPES:
        options["level2_group"] = result.metadata.get("level2_group")
        options["level3_group"] = result.metadata.get("level3_group")
    if result.model_type in {
        "mixed_binary_logit_random_slope",
        "mixed_poisson_random_slope",
        "mixed_negative_binomial_random_slope",
    }:
        options["random_slope_variable"] = result.metadata.get("random_slope_variable")
    if result.model_type in NEGATIVE_BINOMIAL_GLMM_MODEL_TYPES:
        options["quadrature_points"] = result.metadata.get("quadrature_points", 7)
    else:
        options["fe_prior_sd"] = result.metadata.get("fe_prior_sd", 2.0)
        options["variance_prior_sd"] = result.metadata.get("variance_prior_sd", 1.0)
    return options


def _fit_glmm_model(
    dataframe: pd.DataFrame,
    *,
    baseline: RegressionResult,
    optimizer: str,
    model_id: str,
) -> RegressionResult:
    group_variable = None
    if baseline.model_type not in THREE_LEVEL_GLMM_MODEL_TYPES:
        group_variable = str(baseline.metadata.get("group_variable", "")).strip() or None

    return fit_regression_by_level(
        dataframe,
        dependent_variable=baseline.dependent_variable,
        independent_variables=baseline.independent_variables,
        measurement_level=_measurement_level(baseline.model_type),
        model_id=model_id,
        model_type=baseline.model_type,
        group_variable=group_variable,
        mixed_effects_options=_fit_options_from_result(baseline, optimizer),
    )


def _fit_glmm_with_optimizer(
    dataframe: pd.DataFrame,
    *,
    baseline: RegressionResult,
    optimizer: str,
) -> RegressionResult:
    return _fit_glmm_model(
        dataframe,
        baseline=baseline,
        optimizer=optimizer,
        model_id=f"{baseline.model_id}_{optimizer}",
    )


def _direction(estimate: float) -> str:
    if estimate > 0:
        return "positive"
    if estimate < 0:
        return "negative"
    return "zero"


def build_glmm_robustness_report(
    dataframe: pd.DataFrame,
    *,
    baseline_result: RegressionResult,
    optimizers: tuple[str, ...] = ("BFGS",),
    alpha: float = 0.05,
) -> GLMMRobustnessReport:
    """Refit a non-Gaussian GLMM across optimizers and summarize coefficient stability."""
    if baseline_result.model_type not in GLMM_MODEL_TYPES:
        raise ValueError(f"Unsupported GLMM robustness model type: {baseline_result.model_type}")
    if not optimizers:
        raise ValueError("At least one optimizer must be provided.")

    results: list[tuple[str, RegressionResult]] = []
    failed: list[str] = []
    for optimizer in optimizers:
        try:
            results.append(
                (
                    optimizer,
                    _fit_glmm_with_optimizer(
                        dataframe,
                        baseline=baseline_result,
                        optimizer=optimizer,
                    ),
                )
            )
        except (KeyError, ValueError, RuntimeError, FloatingPointError):
            failed.append(optimizer)

    if not results:
        raise ValueError("All GLMM robustness refits failed.")

    comparisons: list[GLMMCoefficientComparison] = []
    for optimizer, result in results:
        for coefficient in result.coefficients:
            comparisons.append(
                GLMMCoefficientComparison(
                    term=coefficient.term,
                    optimizer=optimizer,
                    estimate=coefficient.estimate,
                    standard_error=coefficient.standard_error,
                    statistic=coefficient.statistic,
                    p_value=coefficient.p_value,
                    confidence_interval_lower=coefficient.confidence_interval_lower,
                    confidence_interval_upper=coefficient.confidence_interval_upper,
                    significant=coefficient.p_value < alpha,
                    direction=_direction(coefficient.estimate),
                    converged=result.converged,
                )
            )

    frame = pd.DataFrame([asdict(item) for item in comparisons])
    stability: list[GLMMTermStability] = []
    warnings: list[str] = []
    for term, group in frame.groupby("term", sort=False):
        direction_consistent = group["direction"].nunique() == 1
        significance_consistent = group["significant"].nunique() == 1
        converged_model_count = int(group["converged"].sum())
        model_count = len(group)
        if direction_consistent and significance_consistent and converged_model_count == model_count:
            status = "STABLE"
            interpretation = "Coefficient direction, significance, and convergence were stable."
        elif direction_consistent:
            status = "PARTIALLY_STABLE"
            interpretation = "Coefficient direction was stable, but significance or convergence varied."
        else:
            status = "UNSTABLE"
            interpretation = "Coefficient direction varied across optimizer refits."
        stability.append(
            GLMMTermStability(
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
                "aic": result.fit_statistics.get("aic"),
                "log_likelihood": result.fit_statistics.get("log_likelihood"),
                "group_count": result.fit_statistics.get("group_count"),
                "level2_group_count": result.fit_statistics.get("level2_group_count"),
                "level3_group_count": result.fit_statistics.get("level3_group_count"),
                "random_intercept_variance": result.fit_statistics.get("random_intercept_variance"),
                "random_slope_variance": result.fit_statistics.get("random_slope_variance"),
                "dispersion_alpha": result.fit_statistics.get("dispersion_alpha"),
                "warning_count": len(result.warnings),
            }
            for optimizer, result in results
        ]
    )
    if failed:
        warnings.append("Failed GLMM optimizer refits: " + ", ".join(failed))

    substantive = [item for item in stability if item.term.lower() not in {"const", "intercept"}]
    stable_count = sum(item.status == "STABLE" for item in substantive)
    summary = {
        "model_id": baseline_result.model_id,
        "model_type": baseline_result.model_type,
        "requested_optimizer_count": len(optimizers),
        "successful_optimizer_count": len(results),
        "failed_optimizer_count": len(failed),
        "converged_optimizer_count": int(model_statistics["converged"].sum()),
        "term_count": len(substantive),
        "stable_term_count": stable_count,
        "all_substantive_terms_stable": stable_count == len(substantive) if substantive else True,
    }

    return GLMMRobustnessReport(
        model_id=baseline_result.model_id,
        model_type=baseline_result.model_type,
        dependent_variable=baseline_result.dependent_variable,
        independent_variables=baseline_result.independent_variables,
        optimizers=list(optimizers),
        coefficient_comparisons=comparisons,
        term_stability=stability,
        model_statistics=model_statistics,
        warnings=warnings,
        summary=summary,
    )


def glmm_coefficient_comparison_to_dataframe(report: GLMMRobustnessReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.coefficient_comparisons])


def glmm_stability_summary_to_dataframe(report: GLMMRobustnessReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.term_stability])


def glmm_model_comparison_to_dataframe(report: GLMMRobustnessReport) -> pd.DataFrame:
    return report.model_statistics.copy()


def glmm_robustness_summary_to_dataframe(report: GLMMRobustnessReport) -> pd.DataFrame:
    values = {**report.summary, "warning_count": len(report.warnings)}
    return pd.DataFrame({"item": list(values), "value": list(values.values())})


@dataclass(slots=True)
class GLMMResampledCoefficient:
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
class GLMMAdvancedRobustnessReport:
    model_id: str
    model_type: str
    dependent_variable: str
    independent_variables: list[str]
    group_variable: str
    group_count: int
    coefficients: list[GLMMResampledCoefficient]
    leave_one_group_out: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _glmm_group_columns(result: RegressionResult) -> tuple[str, str | None]:
    if result.model_type in THREE_LEVEL_GLMM_MODEL_TYPES:
        level2_group = str(result.metadata.get("level2_group", "")).strip()
        level3_group = str(result.metadata.get("level3_group", "")).strip()
        if not level2_group or not level3_group:
            raise ValueError("Three-level GLMM robustness requires level2_group and level3_group.")
        return level3_group, level2_group
    group_variable = str(result.metadata.get("group_variable", "")).strip()
    if not group_variable:
        raise ValueError("GLMM robustness requires group_variable metadata.")
    return group_variable, None


def _complete_glmm_data(dataframe: pd.DataFrame, result: RegressionResult) -> pd.DataFrame:
    primary_group, nested_group = _glmm_group_columns(result)
    columns = [result.dependent_variable, *result.independent_variables, primary_group]
    if nested_group is not None:
        columns.append(nested_group)
    frame = dataframe[list(dict.fromkeys(columns))].copy()
    for column in [result.dependent_variable, *result.independent_variables]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna()
    if frame[primary_group].nunique() < 3:
        raise ValueError("Advanced GLMM robustness requires at least 3 groups.")
    return frame


def _resample_glmm_groups(
    frame: pd.DataFrame,
    *,
    primary_group: str,
    nested_group: str | None,
    sampled_groups: np.ndarray,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for position, selected_group in enumerate(sampled_groups):
        piece = frame.loc[frame[primary_group] == selected_group].copy()
        piece[primary_group] = f"{position}_{selected_group}"
        if nested_group is not None:
            piece[nested_group] = piece[nested_group].map(lambda value, prefix=position: f"{prefix}_{value}")
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def build_glmm_advanced_robustness_report(
    dataframe: pd.DataFrame,
    *,
    baseline_result: RegressionResult,
    bootstrap_replications: int = 200,
    run_leave_one_group_out: bool = True,
    confidence_level: float = 0.95,
    random_seed: int = 2026,
    optimizer: str | None = None,
) -> GLMMAdvancedRobustnessReport:
    """Run group bootstrap and leave-one-group-out checks for non-Gaussian GLMMs."""
    if baseline_result.model_type not in GLMM_MODEL_TYPES:
        raise ValueError(f"Unsupported GLMM robustness model type: {baseline_result.model_type}")
    if bootstrap_replications < 5:
        raise ValueError("GLMM group bootstrap requires at least 5 replications.")

    frame = _complete_glmm_data(dataframe, baseline_result)
    primary_group, nested_group = _glmm_group_columns(baseline_result)
    groups = np.asarray(list(frame[primary_group].drop_duplicates()), dtype=object)
    selected_optimizer = optimizer or str(baseline_result.metadata.get("optimizer", "BFGS"))
    baseline_by_term = {item.term: item for item in baseline_result.coefficients}
    rng = np.random.default_rng(random_seed)
    draws: dict[str, list[tuple[float, float]]] = {term: [] for term in baseline_by_term}
    failures = 0

    for replication in range(bootstrap_replications):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sample = _resample_glmm_groups(
            frame,
            primary_group=primary_group,
            nested_group=nested_group,
            sampled_groups=sampled_groups,
        )
        try:
            result = _fit_glmm_model(
                sample,
                baseline=baseline_result,
                optimizer=selected_optimizer,
                model_id=f"{baseline_result.model_id}_group_bootstrap_{replication}",
            )
        except (KeyError, ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
            failures += 1
            continue
        for coefficient in result.coefficients:
            if coefficient.term in draws:
                draws[coefficient.term].append((coefficient.estimate, coefficient.p_value))

    alpha = 1.0 - confidence_level
    coefficients: list[GLMMResampledCoefficient] = []
    for term, values in draws.items():
        if not values:
            continue
        estimates = np.asarray([value[0] for value in values], dtype=float)
        p_values = np.asarray([value[1] for value in values], dtype=float)
        baseline_coefficient = baseline_by_term[term]
        baseline_direction = np.sign(baseline_coefficient.estimate)
        baseline_significant = baseline_coefficient.p_value < alpha
        coefficients.append(
            GLMMResampledCoefficient(
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
            reduced = frame.loc[frame[primary_group] != omitted_group]
            try:
                result = _fit_glmm_model(
                    reduced,
                    baseline=baseline_result,
                    optimizer=selected_optimizer,
                    model_id=f"{baseline_result.model_id}_without_{omitted_group}",
                )
            except (KeyError, ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
                logo_failures += 1
                continue
            for coefficient in result.coefficients:
                logo_rows.append(
                    {
                        "omitted_group": omitted_group,
                        "group_variable": primary_group,
                        "term": coefficient.term,
                        "estimate": coefficient.estimate,
                        "standard_error": coefficient.standard_error,
                        "p_value": coefficient.p_value,
                        "converged": result.converged,
                        "group_count": result.fit_statistics.get("group_count"),
                        "level2_group_count": result.fit_statistics.get("level2_group_count"),
                        "level3_group_count": result.fit_statistics.get("level3_group_count"),
                        "random_intercept_variance": result.fit_statistics.get(
                            "random_intercept_variance"
                        ),
                        "random_slope_variance": result.fit_statistics.get(
                            "random_slope_variance"
                        ),
                        "dispersion_alpha": result.fit_statistics.get("dispersion_alpha"),
                    }
                )

    warnings: list[str] = []
    success_rate = (bootstrap_replications - failures) / bootstrap_replications
    if success_rate < 0.8:
        warnings.append(f"GLMM group bootstrap success rate was {success_rate:.1%}.")
    if logo_failures:
        warnings.append(f"Leave-one-group-out GLMM refits failed for {logo_failures} groups.")
    for item in coefficients:
        if item.term.lower() not in {"const", "intercept"} and item.direction_consistency < 0.9:
            warnings.append(f"{item.term}: bootstrap coefficient direction was unstable.")

    return GLMMAdvancedRobustnessReport(
        model_id=baseline_result.model_id,
        model_type=baseline_result.model_type,
        dependent_variable=baseline_result.dependent_variable,
        independent_variables=baseline_result.independent_variables,
        group_variable=primary_group,
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
            "optimizer": selected_optimizer,
            "nested_group_variable": nested_group,
        },
    )


def glmm_resampling_to_dataframe(report: GLMMAdvancedRobustnessReport) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in report.coefficients])


def glmm_advanced_summary_to_dataframe(report: GLMMAdvancedRobustnessReport) -> pd.DataFrame:
    values = {
        "model_id": report.model_id,
        "model_type": report.model_type,
        "group_variable": report.group_variable,
        "group_count": report.group_count,
        "coefficient_count": len(report.coefficients),
        "warning_count": len(report.warnings),
        **report.metadata,
    }
    return pd.DataFrame({"item": list(values), "value": list(values.values())})
