"""계수형 종속변수 회귀모형 자동선택."""

from __future__ import annotations

import math
import warnings as python_warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.statistics.regression.base import RegressionResult
from src.statistics.regression.negative_binomial import (
    fit_negative_binomial,
)
from src.statistics.regression.poisson import fit_poisson
from src.statistics.regression.zero_inflated_negative_binomial import (
    fit_zero_inflated_negative_binomial,
)
from src.statistics.regression.zero_inflated_poisson import (
    fit_zero_inflated_poisson,
)


def _fit_with_captured_warnings(
    fitter: Callable[..., RegressionResult],
    dataframe: pd.DataFrame,
    **kwargs: object,
) -> tuple[
    RegressionResult,
    list[dict[str, str]],
]:
    """모형 적합 경고를 외부로 노출하지 않고 수집한다."""
    with python_warnings.catch_warnings(record=True) as captured:
        python_warnings.simplefilter("always")
        result = fitter(
            dataframe,
            **kwargs,
        )

    warning_metadata: list[dict[str, str]] = []

    for record in captured:
        item = {
            "category": record.category.__name__,
            "message": str(record.message),
        }
        if item not in warning_metadata:
            warning_metadata.append(item)

    if warning_metadata:
        existing = list(
            result.metadata.get(
                "optimization_warnings",
                [],
            )
        )
        for item in warning_metadata:
            if item not in existing:
                existing.append(item)

        result.metadata["optimization_warnings"] = existing
        result.metadata["optimization_warning_count"] = len(existing)

    return result, warning_metadata


def _predicted_zero_proportion(
    result: RegressionResult,
) -> float:
    fitted = result.raw_result

    if fitted is None:
        raise ValueError("원본 statsmodels 결과 객체가 없습니다.")

    if result.model_type in {
        "zero_inflated_poisson",
        "zero_inflated_negative_binomial",
    }:
        return float(np.mean(fitted.predict(which="prob-zero")))

    predicted = np.asarray(
        fitted.predict(),
        dtype=float,
    )

    if result.model_type == "negative_binomial":
        alpha = float(result.fit_statistics["alpha"])
        return float(np.mean((1 + alpha * predicted) ** (-1 / alpha)))

    return float(np.mean(np.exp(-predicted)))


def _safe_candidate(
    fitter: Callable[..., RegressionResult],
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None,
    model_id: str,
    covariance_type: str,
    add_intercept: bool,
    maximum_iterations: int,
) -> tuple[
    RegressionResult | None,
    str | None,
]:
    try:
        with python_warnings.catch_warnings(record=True) as captured:
            python_warnings.simplefilter("always")

            result = fitter(
                dataframe,
                dependent_variable=dependent_variable,
                independent_variables=independent_variables,
                fixed_effects=fixed_effects,
                model_id=model_id,
                covariance_type=covariance_type,
                add_intercept=add_intercept,
                maximum_iterations=maximum_iterations,
            )
    except (
        ValueError,
        RuntimeError,
        np.linalg.LinAlgError,
        FloatingPointError,
    ) as error:
        return None, str(error)

    leaked_warnings: list[dict[str, str]] = []

    for record in captured:
        item = {
            "category": record.category.__name__,
            "message": str(record.message),
        }
        if item not in leaked_warnings:
            leaked_warnings.append(item)

    if leaked_warnings:
        existing = list(
            result.metadata.get(
                "optimization_warnings",
                [],
            )
        )

        for item in leaked_warnings:
            if item not in existing:
                existing.append(item)

        result.metadata["optimization_warnings"] = existing
        result.metadata["optimization_warning_count"] = len(existing)

    if not result.converged:
        return None, (f"{result.model_type} 모형이 수렴하지 않았습니다.")

    aic = float(result.fit_statistics["aic"])
    bic = float(result.fit_statistics["bic"])
    if not (np.isfinite(aic) and np.isfinite(bic)):
        return None, (f"{result.model_type} 모형의 AIC 또는 BIC가 유한하지 않습니다.")

    coefficient_values = np.asarray(
        [
            value
            for coefficient in result.coefficients
            for value in (
                coefficient.estimate,
                coefficient.standard_error,
                coefficient.p_value,
                coefficient.confidence_interval_lower,
                coefficient.confidence_interval_upper,
            )
        ],
        dtype=float,
    )
    if not np.isfinite(coefficient_values).all():
        return None, (f"{result.model_type} 모형에 비유한 계수 또는 표준오차가 있습니다.")

    invalid_warning_categories = {
        "ConvergenceWarning",
        "HessianInversionWarning",
    }
    warning_categories = {
        item.get("category", "")
        for item in result.metadata.get(
            "optimization_warnings",
            [],
        )
    }
    invalid_categories = sorted(warning_categories & invalid_warning_categories)
    if invalid_categories:
        return None, (
            f"{result.model_type} 모형에서 " + ", ".join(invalid_categories) + "이 기록되었습니다."
        )

    return result, None


def fit_count_regression(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "count_1",
    covariance_type: str = "HC3",
    add_intercept: bool = True,
    maximum_iterations: int = 500,
    dispersion_threshold: float = 1.5,
    zero_excess_threshold: float = 0.10,
    minimum_aic_improvement: float = 2.0,
) -> RegressionResult:
    """
    Poisson, NB2, ZIP, ZINB 가운데 적합한 계수형 모형을 선택한다.

    1. Poisson Pearson 분산비로 Poisson/NB2 기본모형을 선택한다.
    2. 관측 0 비율이 기본모형의 예측 0 비율보다 충분히 높으면
       ZIP과 필요 시 ZINB를 후보로 적합한다.
    3. 수렴한 후보가 기본모형보다 AIC를 최소 기준 이상 개선할 때만
       영과잉 모형으로 전환한다.
    """
    if dispersion_threshold <= 1:
        raise ValueError("과산포 자동선택 기준은 1보다 커야 합니다.")

    if not 0 <= zero_excess_threshold < 1:
        raise ValueError("영과잉 자동선택 기준은 0 이상 1 미만이어야 합니다.")

    if minimum_aic_improvement < 0:
        raise ValueError("최소 AIC 개선 기준은 0 이상이어야 합니다.")

    poisson, poisson_fit_warnings = _fit_with_captured_warnings(
        fit_poisson,
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_id=model_id,
        covariance_type=covariance_type,
        add_intercept=add_intercept,
        maximum_iterations=maximum_iterations,
    )

    dispersion = float(poisson.fit_statistics["dispersion_ratio"])

    baseline = poisson
    negative_binomial_fitted = False
    negative_binomial_fit_warnings: list[dict[str, str]] = []

    if math.isfinite(dispersion) and dispersion > dispersion_threshold:
        (
            negative_binomial,
            negative_binomial_fit_warnings,
        ) = _fit_with_captured_warnings(
            fit_negative_binomial,
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            covariance_type=covariance_type,
            add_intercept=add_intercept,
            maximum_iterations=maximum_iterations,
        )
        negative_binomial_fitted = True
        baseline = negative_binomial

    observed_zero_proportion = float(baseline.fit_statistics["zero_proportion"])
    baseline_predicted_zero = _predicted_zero_proportion(baseline)
    zero_excess = float(observed_zero_proportion - baseline_predicted_zero)

    common_metadata = {
        "count_model_selection_method": ("dispersion_then_zero_inflation_aic"),
        "dispersion_threshold": dispersion_threshold,
        "zero_excess_threshold": zero_excess_threshold,
        "minimum_aic_improvement": (minimum_aic_improvement),
        "poisson_dispersion_ratio": dispersion,
        "poisson_aic": float(poisson.fit_statistics["aic"]),
        "poisson_bic": float(poisson.fit_statistics["bic"]),
        "negative_binomial_fitted": (negative_binomial_fitted),
        "poisson_fit_warnings": poisson_fit_warnings,
        "negative_binomial_fit_warnings": (negative_binomial_fit_warnings),
        "baseline_count_model": (baseline.model_type),
        "baseline_aic": float(baseline.fit_statistics["aic"]),
        "baseline_bic": float(baseline.fit_statistics["bic"]),
        "observed_zero_proportion": (observed_zero_proportion),
        "baseline_predicted_zero_proportion": (baseline_predicted_zero),
        "zero_excess": zero_excess,
    }

    if negative_binomial_fitted:
        common_metadata.update(
            {
                "negative_binomial_aic": float(baseline.fit_statistics["aic"]),
                "negative_binomial_bic": float(baseline.fit_statistics["bic"]),
                "negative_binomial_alpha": float(baseline.fit_statistics["alpha"]),
            }
        )

    should_fit_zero_inflated = (
        zero_excess > zero_excess_threshold or observed_zero_proportion >= 0.30
    )

    if not should_fit_zero_inflated:
        baseline.metadata.update(
            {
                **common_metadata,
                "selected_count_model": baseline.model_type,
                "zero_inflated_candidates_fitted": False,
                "selection_reason": (
                    "관측 0 비율과 기본모형의 예측 0 비율이 "
                    "영과잉 후보 적합 기준을 충족하지 않아 "
                    "기본모형을 유지했습니다."
                ),
            }
        )
        return baseline

    candidates: list[RegressionResult] = [baseline]
    candidate_errors: list[str] = []

    zip_result, zip_error = _safe_candidate(
        fit_zero_inflated_poisson,
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_id=model_id,
        covariance_type=covariance_type,
        add_intercept=add_intercept,
        maximum_iterations=maximum_iterations,
    )
    if zip_result is not None and zip_result.converged:
        candidates.append(zip_result)
    if zip_error is not None:
        candidate_errors.append(f"ZIP: {zip_error}")

    zinb_result: RegressionResult | None = None
    if negative_binomial_fitted or (
        math.isfinite(dispersion) and dispersion > dispersion_threshold
    ):
        zinb_result, zinb_error = _safe_candidate(
            fit_zero_inflated_negative_binomial,
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            covariance_type=covariance_type,
            add_intercept=add_intercept,
            maximum_iterations=maximum_iterations,
        )
        if zinb_result is not None and zinb_result.converged:
            candidates.append(zinb_result)
        if zinb_error is not None:
            candidate_errors.append(f"ZINB: {zinb_error}")

    selected = min(
        candidates,
        key=lambda result: float(result.fit_statistics["aic"]),
    )

    baseline_aic = float(baseline.fit_statistics["aic"])
    selected_aic = float(selected.fit_statistics["aic"])
    aic_improvement = float(baseline_aic - selected_aic)

    if selected is baseline or aic_improvement < minimum_aic_improvement:
        selected = baseline
        selection_reason = (
            "영과잉 후보가 기본모형보다 AIC를 충분히 개선하지 못해 기본모형을 유지했습니다."
        )
    else:
        selection_reason = (
            "관측 0 비율이 기본모형 예측보다 높고 영과잉 "
            "후보가 AIC를 충분히 개선하여 영과잉 모형을 선택했습니다."
        )
        selected.warnings.insert(
            0,
            (
                "영과잉 가능성과 AIC 개선이 확인되어 "
                f"{selected.model_type} 모형으로 자동 전환했습니다."
            ),
        )

    candidate_aic = {
        result.model_type: float(result.fit_statistics["aic"]) for result in candidates
    }
    candidate_bic = {
        result.model_type: float(result.fit_statistics["bic"]) for result in candidates
    }

    selected.metadata.update(
        {
            **common_metadata,
            "selected_count_model": (selected.model_type),
            "zero_inflated_candidates_fitted": (True),
            "candidate_aic": candidate_aic,
            "candidate_bic": candidate_bic,
            "candidate_errors": candidate_errors,
            "selected_aic": float(selected.fit_statistics["aic"]),
            "selected_bic": float(selected.fit_statistics["bic"]),
            "aic_improvement_over_baseline": (baseline_aic - float(selected.fit_statistics["aic"])),
            "selection_reason": selection_reason,
        }
    )

    return selected
