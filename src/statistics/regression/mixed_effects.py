"""Random Intercept 및 Random Slope 선형 혼합효과 회귀모형 구현."""

from __future__ import annotations

import math
import warnings as python_warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM, MixedLMParams

from src.statistics.regression.base import (
    ModelCoefficient,
    RegressionResult,
    validate_model_variables,
)

SUPPORTED_OPTIMIZERS = {"bfgs", "cg", "lbfgs", "nm", "powell"}


def fit_random_intercept(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    group_variable: str,
    model_id: str = "mixed_random_intercept_1",
    add_intercept: bool = True,
    reml: bool = False,
    method: str = "lbfgs",
    max_iterations: int = 200,
    random_effect_covariance: str = "correlated",
) -> RegressionResult:
    """Random Intercept 선형 혼합효과 모형을 적합한다."""
    return _fit_mixed_effects(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        group_variable=group_variable,
        random_slope_variable=None,
        random_slope_variables=None,
        model_id=model_id,
        add_intercept=add_intercept,
        reml=reml,
        method=method,
        max_iterations=max_iterations,
        cross_level_predictor=None,
        cross_level_moderator=None,
        level1_centering="none",
        level2_centering="none",
        simple_slope_values=None,
        johnson_neyman=False,
        random_effect_covariance=random_effect_covariance,
    )


def fit_random_slope(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    group_variable: str,
    random_slope_variable: str,
    model_id: str = "mixed_random_slope_1",
    add_intercept: bool = True,
    reml: bool = False,
    method: str = "lbfgs",
    max_iterations: int = 200,
    cross_level_predictor: str | None = None,
    cross_level_moderator: str | None = None,
    level1_centering: str = "none",
    level2_centering: str = "none",
    simple_slope_values: list[str | float] | None = None,
    johnson_neyman: bool = False,
    random_effect_covariance: str = "correlated",
) -> RegressionResult:
    """상관된 Random Intercept와 Random Slope 모형을 적합한다.

    predictor와 moderator가 함께 지정되면 교차수준 상호작용을 자동 생성한다.
    """
    return _fit_mixed_effects(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        group_variable=group_variable,
        random_slope_variable=random_slope_variable,
        random_slope_variables=[random_slope_variable],
        model_id=model_id,
        add_intercept=add_intercept,
        reml=reml,
        method=method,
        max_iterations=max_iterations,
        cross_level_predictor=cross_level_predictor,
        cross_level_moderator=cross_level_moderator,
        level1_centering=level1_centering,
        level2_centering=level2_centering,
        simple_slope_values=simple_slope_values,
        johnson_neyman=johnson_neyman,
        random_effect_covariance=random_effect_covariance,
    )


def fit_multiple_random_slopes(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    group_variable: str,
    random_slope_variables: list[str],
    model_id: str = "mixed_random_slope_1",
    add_intercept: bool = True,
    reml: bool = False,
    method: str = "lbfgs",
    max_iterations: int = 200,
    random_effect_covariance: str = "correlated",
) -> RegressionResult:
    """하나 이상의 Random Slope 모형을 적합한다."""
    slopes = list(dict.fromkeys(str(v).strip() for v in random_slope_variables if str(v).strip()))
    if not slopes:
        raise ValueError("Random Slope 변수를 하나 이상 지정해야 합니다.")
    return _fit_mixed_effects(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        group_variable=group_variable,
        random_slope_variable=slopes[0],
        random_slope_variables=slopes,
        model_id=model_id,
        add_intercept=add_intercept,
        reml=reml,
        method=method,
        max_iterations=max_iterations,
        cross_level_predictor=None,
        cross_level_moderator=None,
        level1_centering="none",
        level2_centering="none",
        simple_slope_values=None,
        johnson_neyman=False,
        random_effect_covariance=random_effect_covariance,
    )


def _fit_mixed_effects(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    group_variable: str,
    random_slope_variable: str | None,
    random_slope_variables: list[str] | None,
    model_id: str,
    add_intercept: bool,
    reml: bool,
    method: str,
    max_iterations: int,
    cross_level_predictor: str | None,
    cross_level_moderator: str | None,
    level1_centering: str,
    level2_centering: str,
    simple_slope_values: list[str | float] | None,
    johnson_neyman: bool,
    random_effect_covariance: str,
) -> RegressionResult:
    independent_variables = list(dict.fromkeys(independent_variables))
    random_effect_covariance = str(random_effect_covariance).strip().lower()
    if random_effect_covariance not in {"correlated", "diagonal"}:
        raise ValueError("random_effect_covariance는 correlated 또는 diagonal이어야 합니다.")
    random_slope_variables = list(
        dict.fromkeys(
            random_slope_variables or ([random_slope_variable] if random_slope_variable else [])
        )
    )
    random_slope_variable = random_slope_variables[0] if random_slope_variables else None
    working_dataframe = dataframe.copy()
    cross_level_metadata: dict[str, Any] | None = None
    if cross_level_predictor is not None or cross_level_moderator is not None:
        if not cross_level_predictor or not cross_level_moderator:
            raise ValueError("교차수준 상호작용에는 predictor와 moderator를 모두 지정해야 합니다.")
        working_dataframe, independent_variables, random_slope_variable, cross_level_metadata = (
            _prepare_cross_level_interaction(
                working_dataframe,
                independent_variables=independent_variables,
                group_variable=group_variable,
                random_slope_variable=random_slope_variable,
                predictor=cross_level_predictor,
                moderator=cross_level_moderator,
                level1_centering=level1_centering,
                level2_centering=level2_centering,
            )
        )
        random_slope_variables = [random_slope_variable]
    _validate_inputs(
        working_dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        group_variable=group_variable,
        random_slope_variable=random_slope_variable,
        random_slope_variables=random_slope_variables,
        method=method,
        max_iterations=max_iterations,
    )
    complete = _prepare_data(
        working_dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        group_variable=group_variable,
    )
    outcome = complete[dependent_variable].astype(float)
    predictors = complete[independent_variables].astype(float).copy()
    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")
    _validate_design_matrix(predictors)

    groups = complete[group_variable]
    group_sizes = groups.value_counts()
    random_design = None
    if random_slope_variables:
        random_design = complete[random_slope_variables].astype(float).copy()
        if add_intercept:
            random_design = sm.add_constant(random_design, has_constant="add")

    model = MixedLM(endog=outcome, exog=predictors, groups=groups, exog_re=random_design)
    warning_messages: list[str] = []
    try:
        with python_warnings.catch_warnings(record=True) as caught:
            python_warnings.simplefilter("always")
            fit_kwargs = {}
            if random_design is not None and random_effect_covariance == "diagonal":
                free = MixedLMParams.from_components(
                    fe_params=np.ones(predictors.shape[1]),
                    cov_re=np.eye(random_design.shape[1]),
                )
                fit_kwargs["free"] = free
            fitted = model.fit(
                reml=reml,
                method=method,
                maxiter=max_iterations,
                disp=False,
                **fit_kwargs,
            )
    except (np.linalg.LinAlgError, ValueError) as error:
        raise ValueError(
            "혼합효과 모형을 추정할 수 없습니다. 설계행렬, 그룹 구조, 랜덤효과 구조 또는 초기값을 확인하세요."
        ) from error
    for item in caught:
        _append_unique_warning(warning_messages, str(item.message))

    confidence_intervals = fitted.conf_int()
    coefficients = [
        ModelCoefficient(
            term=str(term),
            estimate=float(fitted.fe_params[term]),
            standard_error=float(fitted.bse_fe[term]),
            statistic=float(fitted.tvalues[term]),
            p_value=float(fitted.pvalues[term]),
            confidence_interval_lower=float(confidence_intervals.loc[term, 0]),
            confidence_interval_upper=float(confidence_intervals.loc[term, 1]),
        )
        for term in fitted.fe_params.index
    ]

    cov_re = np.asarray(fitted.cov_re, dtype=float)
    intercept_variance = float(cov_re[0, 0])
    residual_variance = float(fitted.scale)
    fit_statistics: dict[str, Any] = {
        "log_likelihood": float(fitted.llf),
        "aic": _finite_float_or_none(fitted.aic),
        "bic": _finite_float_or_none(fitted.bic),
        "random_intercept_variance": intercept_variance,
        "residual_variance": residual_variance,
        "intraclass_correlation": (
            intercept_variance / (intercept_variance + residual_variance)
            if intercept_variance + residual_variance > 0
            else None
        ),
        "group_count": int(groups.nunique()),
        "minimum_group_size": int(group_sizes.min()),
        "maximum_group_size": int(group_sizes.max()),
        "singleton_group_count": int((group_sizes == 1).sum()),
    }
    if random_slope_variables:
        eigenvalues = np.linalg.eigvalsh(cov_re)
        variances = {
            term: float(cov_re[i + 1, i + 1]) for i, term in enumerate(random_slope_variables)
        }
        covariance_matrix = {
            row: {col: float(cov_re[i + 1, j + 1]) for j, col in enumerate(random_slope_variables)}
            for i, row in enumerate(random_slope_variables)
        }
        intercept_covariances = {
            term: float(cov_re[0, i + 1]) for i, term in enumerate(random_slope_variables)
        }
        correlations = {}
        for i, left in enumerate(["intercept", *random_slope_variables]):
            correlations[left] = {}
            for j, right in enumerate(["intercept", *random_slope_variables]):
                denom = math.sqrt(max(float(cov_re[i, i]), 0.0) * max(float(cov_re[j, j]), 0.0))
                correlations[left][right] = float(cov_re[i, j] / denom) if denom > 0 else None
        fit_statistics.update(
            {
                "random_slope_variances": variances,
                "random_slope_covariance_matrix": covariance_matrix,
                "random_intercept_slope_covariances": intercept_covariances,
                "random_effect_correlation_matrix": correlations,
                "random_effect_covariance_min_eigenvalue": float(eigenvalues.min()),
                "random_effect_covariance_determinant": float(np.linalg.det(cov_re)),
            }
        )
        first = random_slope_variables[0]
        fit_statistics.update(
            {
                "random_slope_variance": variances[first],
                "random_intercept_slope_covariance": intercept_covariances[first],
                "random_intercept_slope_correlation": correlations["intercept"][first],
            }
        )
        near_zero = [term for term, value in variances.items() if value <= 1e-8]
        if near_zero:
            _append_unique_warning(
                warning_messages,
                "Random Slope 분산이 0에 가까운 변수가 있습니다: " + ", ".join(near_zero),
            )
        if eigenvalues.min() <= 1e-8:
            _append_unique_warning(
                warning_messages, "랜덤효과 공분산행렬이 특이하거나 거의 특이합니다."
            )
        extreme = [
            (a, b)
            for a, row in correlations.items()
            for b, value in row.items()
            if a < b and value is not None and abs(value) >= 0.995
        ]
        if extreme and random_effect_covariance == "correlated":
            _append_unique_warning(
                warning_messages, "랜덤효과 상관의 절대값이 1에 가까워 특이 적합 가능성이 있습니다."
            )

    converged = bool(fitted.converged)
    if not converged:
        _append_unique_warning(warning_messages, "혼합효과 모형이 수렴하지 않았습니다.")
    if len(group_sizes) < 5:
        _append_unique_warning(
            warning_messages, "그룹 수가 5개 미만이어서 랜덤효과 분산 추정이 불안정할 수 있습니다."
        )
    if int((group_sizes == 1).sum()) > 0:
        _append_unique_warning(
            warning_messages, f"관측치가 1개뿐인 그룹이 {int((group_sizes == 1).sum())}개 있습니다."
        )
    if intercept_variance <= np.finfo(float).eps:
        _append_unique_warning(warning_messages, "Random Intercept 분산이 0에 가깝습니다.")

    if cross_level_metadata is not None:
        interaction_term = cross_level_metadata["interaction_term"]
        conditional_effects = _calculate_conditional_effects(
            fitted,
            predictor_term=cross_level_metadata["predictor_term"],
            moderator_term=cross_level_metadata["moderator_term"],
            interaction_term=interaction_term,
            moderator_mean=float(cross_level_metadata["moderator_mean_centered"]),
            moderator_sd=float(cross_level_metadata["moderator_sd"]),
            requested_values=simple_slope_values,
        )
        cross_level_metadata["conditional_effects"] = conditional_effects
        cross_level_metadata["johnson_neyman"] = (
            _calculate_johnson_neyman(
                fitted,
                predictor_term=cross_level_metadata["predictor_term"],
                interaction_term=interaction_term,
                moderator_min=float(cross_level_metadata["moderator_min_centered"]),
                moderator_max=float(cross_level_metadata["moderator_max_centered"]),
            )
            if johnson_neyman
            else None
        )
        interaction = next((item for item in coefficients if item.term == interaction_term), None)
        if interaction is not None:
            fit_statistics["cross_level_interaction_estimate"] = interaction.estimate
            fit_statistics["cross_level_interaction_p_value"] = interaction.p_value

    model_type = "mixed_random_slope" if random_slope_variables else "mixed_random_intercept"
    return RegressionResult(
        model_id=model_id,
        model_type=model_type,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(fitted.nobs),
        coefficients=coefficients,
        fit_statistics=fit_statistics,
        converged=converged,
        standard_error_type="model_based",
        warnings=warning_messages,
        metadata={
            "group_variable": group_variable,
            "random_slope_variable": random_slope_variable,
            "random_slope_variables": random_slope_variables,
            "random_effect_terms": ["intercept", *random_slope_variables],
            "random_effect_covariance": random_effect_covariance,
            "random_effects_correlated": random_effect_covariance == "correlated",
            "add_intercept": add_intercept,
            "reml": reml,
            "optimizer": method,
            "max_iterations": max_iterations,
            "dropped_case_count": len(working_dataframe) - len(complete),
            "design_matrix_columns": [str(column) for column in predictors.columns],
            "cross_level_interaction": cross_level_metadata,
        },
        raw_result=fitted,
    )


def _validate_inputs(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    group_variable: str,
    random_slope_variable: str | None,
    random_slope_variables: list[str],
    method: str,
    max_iterations: int,
) -> None:
    validate_model_variables(dataframe, dependent_variable, independent_variables)
    if not group_variable.strip():
        raise ValueError("그룹변수는 비어 있을 수 없습니다.")
    if group_variable == dependent_variable or group_variable in independent_variables:
        raise ValueError("그룹변수는 종속변수 또는 독립변수와 중복될 수 없습니다.")
    if group_variable not in dataframe.columns:
        raise KeyError(f"데이터에 그룹변수가 없습니다: {group_variable}")
    for slope in random_slope_variables:
        if slope not in independent_variables:
            raise ValueError("Random Slope 변수는 독립변수에 포함되어야 합니다: " + slope)
        if slope not in dataframe.columns:
            raise KeyError(f"데이터에 Random Slope 변수가 없습니다: {slope}")
    if method not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"지원하지 않는 혼합효과 최적화 방식입니다: {method}")
    if max_iterations <= 0:
        raise ValueError("최대 반복 횟수는 1 이상이어야 합니다.")


def _prepare_data(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    group_variable: str,
) -> pd.DataFrame:
    selected = dataframe[[dependent_variable, *independent_variables, group_variable]].copy()
    for variable in [dependent_variable, *independent_variables]:
        selected[variable] = pd.to_numeric(selected[variable], errors="coerce")
    numeric_columns = [dependent_variable, *independent_variables]
    selected[numeric_columns] = selected[numeric_columns].replace([np.inf, -np.inf], np.nan)
    complete = selected.dropna()
    if complete.empty:
        raise ValueError("혼합효과 회귀분석에 사용할 완전사례가 없습니다.")
    if complete[dependent_variable].nunique() <= 1:
        raise ValueError("종속변수가 상수이거나 유효 범주가 하나뿐입니다.")
    constant_predictors = [v for v in independent_variables if complete[v].nunique() <= 1]
    if constant_predictors:
        raise ValueError("상수 독립변수가 있습니다: " + ", ".join(constant_predictors))
    if complete[group_variable].nunique() <= 1:
        raise ValueError("혼합효과 모형에는 두 개 이상의 그룹이 필요합니다.")
    return complete


def _validate_design_matrix(predictors: pd.DataFrame) -> None:
    if len(predictors) <= len(predictors.columns):
        raise ValueError("표본 수가 고정효과 추정 모수 수보다 많아야 합니다.")
    if int(np.linalg.matrix_rank(predictors.to_numpy(dtype=float))) < len(predictors.columns):
        raise ValueError("고정효과 설계행렬에 완전 다중공선성이 있습니다.")


def _append_unique_warning(messages: list[str], message: str) -> None:
    if message not in messages:
        messages.append(message)


def _finite_float_or_none(value: Any) -> float | None:
    numeric_value = float(value)
    return numeric_value if math.isfinite(numeric_value) else None


CENTERING_METHODS = {"none", "grand_mean", "group_mean"}


def _prepare_cross_level_interaction(
    dataframe: pd.DataFrame,
    *,
    independent_variables: list[str],
    group_variable: str,
    random_slope_variable: str | None,
    predictor: str,
    moderator: str,
    level1_centering: str,
    level2_centering: str,
) -> tuple[pd.DataFrame, list[str], str, dict[str, Any]]:
    if predictor not in dataframe.columns or moderator not in dataframe.columns:
        missing = [v for v in (predictor, moderator) if v not in dataframe.columns]
        raise KeyError("교차수준 상호작용 변수가 없습니다: " + ", ".join(missing))
    if predictor not in independent_variables or moderator not in independent_variables:
        raise ValueError("교차수준 predictor와 moderator는 독립변수에 포함되어야 합니다.")
    if level1_centering not in CENTERING_METHODS or level2_centering not in CENTERING_METHODS:
        raise ValueError("중심화 방식은 none, grand_mean, group_mean 중 하나여야 합니다.")
    if level2_centering == "group_mean":
        raise ValueError("집단수준 moderator에는 group_mean centering을 적용할 수 없습니다.")

    work = dataframe.copy()
    x = pd.to_numeric(work[predictor], errors="coerce")
    z = pd.to_numeric(work[moderator], errors="coerce")
    x_term = predictor
    z_term = moderator
    x_center = 0.0
    z_center = 0.0
    if level1_centering == "grand_mean":
        x_center = float(x.mean())
        x_term = f"{predictor}__grand_mean_centered"
        work[x_term] = x - x_center
    elif level1_centering == "group_mean":
        x_term = f"{predictor}__group_mean_centered"
        group_means = x.groupby(work[group_variable]).transform("mean")
        work[x_term] = x - group_means
    if level2_centering == "grand_mean":
        z_center = float(z.mean())
        z_term = f"{moderator}__grand_mean_centered"
        work[z_term] = z - z_center

    interaction_term = f"{x_term}__by__{z_term}"
    if interaction_term in dataframe.columns:
        raise ValueError(f"자동 생성 상호작용항 이름이 기존 변수와 충돌합니다: {interaction_term}")
    work[interaction_term] = pd.to_numeric(work[x_term], errors="coerce") * pd.to_numeric(
        work[z_term], errors="coerce"
    )
    updated = [v for v in independent_variables if v not in {predictor, moderator}]
    updated.extend([x_term, z_term, interaction_term])
    z_centered = pd.to_numeric(work[z_term], errors="coerce")
    within_group_nunique = work.groupby(group_variable)[moderator].nunique(dropna=True)
    metadata = {
        "predictor": predictor,
        "moderator": moderator,
        "predictor_term": x_term,
        "moderator_term": z_term,
        "interaction_term": interaction_term,
        "level1_centering": level1_centering,
        "level2_centering": level2_centering,
        "predictor_center": x_center,
        "moderator_center": z_center,
        "moderator_mean_centered": float(z_centered.mean()),
        "moderator_sd": float(z_centered.std(ddof=1)),
        "moderator_min_centered": float(z_centered.min()),
        "moderator_max_centered": float(z_centered.max()),
        "moderator_constant_within_group_share": float((within_group_nunique <= 1).mean()),
    }
    return work, list(dict.fromkeys(updated)), x_term, metadata


def _calculate_conditional_effects(
    fitted: Any,
    *,
    predictor_term: str,
    moderator_term: str,
    interaction_term: str,
    moderator_mean: float,
    moderator_sd: float,
    requested_values: list[str | float] | None,
) -> list[dict[str, float | str]]:
    requested = requested_values or ["minus_1_sd", "mean", "plus_1_sd"]
    mapping = {
        "minus_1_sd": moderator_mean - moderator_sd,
        "mean": moderator_mean,
        "plus_1_sd": moderator_mean + moderator_sd,
    }
    covariance = fitted.cov_params()
    b1 = float(fitted.fe_params[predictor_term])
    b3 = float(fitted.fe_params[interaction_term])
    v1 = float(covariance.loc[predictor_term, predictor_term])
    v3 = float(covariance.loc[interaction_term, interaction_term])
    c13 = float(covariance.loc[predictor_term, interaction_term])
    rows = []
    for item in requested:
        value = mapping[item] if isinstance(item, str) and item in mapping else float(item)
        estimate = b1 + b3 * value
        variance = max(v1 + value * value * v3 + 2.0 * value * c13, 0.0)
        se = math.sqrt(variance)
        statistic = estimate / se if se > 0 else math.nan
        p = 2.0 * stats.norm.sf(abs(statistic)) if math.isfinite(statistic) else math.nan
        rows.append(
            {
                "label": str(item),
                "moderator_value": float(value),
                "estimate": estimate,
                "standard_error": se,
                "statistic": statistic,
                "p_value": float(p),
                "confidence_interval_lower": estimate - 1.96 * se,
                "confidence_interval_upper": estimate + 1.96 * se,
            }
        )
    return rows


def _calculate_johnson_neyman(
    fitted: Any,
    *,
    predictor_term: str,
    interaction_term: str,
    moderator_min: float,
    moderator_max: float,
) -> dict[str, Any]:
    covariance = fitted.cov_params()
    b1 = float(fitted.fe_params[predictor_term])
    b3 = float(fitted.fe_params[interaction_term])
    v1 = float(covariance.loc[predictor_term, predictor_term])
    v3 = float(covariance.loc[interaction_term, interaction_term])
    c13 = float(covariance.loc[predictor_term, interaction_term])
    critical = 1.96
    roots = np.roots(
        [
            b3 * b3 - critical**2 * v3,
            2 * b1 * b3 - 2 * critical**2 * c13,
            b1 * b1 - critical**2 * v1,
        ]
    )
    real = sorted(
        float(r.real) for r in roots if abs(r.imag) < 1e-8 and math.isfinite(float(r.real))
    )
    return {
        "critical_value": critical,
        "roots": real,
        "observed_range": [moderator_min, moderator_max],
        "roots_within_observed_range": [r for r in real if moderator_min <= r <= moderator_max],
    }


def fit_three_level_mixed_effects(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    level2_group: str,
    level3_group: str,
    level2_random_slope_variables: list[str] | None = None,
    level3_random_slope_variables: list[str] | None = None,
    model_id: str = "mixed_three_level_1",
    add_intercept: bool = True,
    reml: bool = False,
    method: str = "lbfgs",
    max_iterations: int = 300,
) -> RegressionResult:
    """중첩된 3수준 선형 혼합효과 모형을 적합한다.

    Level 1 관측치가 Level 2 그룹에, Level 2 그룹이 Level 3 그룹에
    완전히 중첩된 구조를 전제로 한다. Level 3 랜덤효과는 상관 구조,
    Level 2 랜덤효과는 독립 variance component로 추정한다.
    """
    independent_variables = list(dict.fromkeys(independent_variables))
    level2_slopes = list(dict.fromkeys(level2_random_slope_variables or []))
    level3_slopes = list(dict.fromkeys(level3_random_slope_variables or []))
    validate_model_variables(dataframe, dependent_variable, independent_variables)
    if level2_group == level3_group:
        raise ValueError("Level 2와 Level 3 그룹변수는 서로 달라야 합니다.")
    missing_groups = [v for v in (level2_group, level3_group) if v not in dataframe.columns]
    if missing_groups:
        raise KeyError("데이터에 그룹변수가 없습니다: " + ", ".join(missing_groups))
    for slope in [*level2_slopes, *level3_slopes]:
        if slope not in independent_variables:
            raise ValueError("3수준 Random Slope 변수는 독립변수에 포함되어야 합니다: " + slope)
    if method not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"지원하지 않는 혼합효과 최적화 방식입니다: {method}")
    if max_iterations <= 0:
        raise ValueError("최대 반복 횟수는 1 이상이어야 합니다.")

    selected = dataframe[
        [dependent_variable, *independent_variables, level2_group, level3_group]
    ].copy()
    for variable in [dependent_variable, *independent_variables]:
        selected[variable] = pd.to_numeric(selected[variable], errors="coerce")
    selected[[dependent_variable, *independent_variables]] = selected[
        [dependent_variable, *independent_variables]
    ].replace([np.inf, -np.inf], np.nan)
    complete = selected.dropna()
    if complete.empty:
        raise ValueError("3수준 혼합효과 분석에 사용할 완전사례가 없습니다.")
    if complete[dependent_variable].nunique() <= 1:
        raise ValueError("종속변수가 상수이거나 유효 범주가 하나뿐입니다.")
    if complete[level3_group].nunique() <= 1 or complete[level2_group].nunique() <= 1:
        raise ValueError("3수준 모형에는 Level 2와 Level 3에 각각 두 개 이상의 그룹이 필요합니다.")

    nesting_counts = complete.groupby(level2_group, observed=True)[level3_group].nunique()
    invalid_level2 = nesting_counts[nesting_counts > 1]
    if not invalid_level2.empty:
        examples = ", ".join(map(str, invalid_level2.index[:5]))
        raise ValueError(
            "Level 2 그룹이 둘 이상의 Level 3 그룹에 속해 완전 중첩 구조가 아닙니다: " + examples
        )

    working = complete.copy()
    working["__level2_nested__"] = (
        working[level3_group].astype(str) + "::" + working[level2_group].astype(str)
    )
    fixed_formula = dependent_variable + " ~ " + " + ".join(independent_variables)
    if not add_intercept:
        fixed_formula += " - 1"
    re_formula = "1"
    if level3_slopes:
        re_formula += " + " + " + ".join(level3_slopes)
    vc_formula: dict[str, str] = {
        "level2_intercept": "0 + C(__level2_nested__)",
    }
    for slope in level2_slopes:
        vc_formula[f"level2_slope:{slope}"] = f"0 + C(__level2_nested__):{slope}"

    model = MixedLM.from_formula(
        fixed_formula,
        data=working,
        groups=working[level3_group],
        re_formula=re_formula,
        vc_formula=vc_formula,
    )
    warning_messages: list[str] = []
    try:
        with python_warnings.catch_warnings(record=True) as caught:
            python_warnings.simplefilter("always")
            fitted = model.fit(
                reml=reml,
                method=method,
                maxiter=max_iterations,
                disp=False,
            )
    except (np.linalg.LinAlgError, ValueError) as error:
        raise ValueError(
            "3수준 혼합효과 모형을 추정할 수 없습니다. 중첩 구조, 그룹 수, 설계행렬 또는 랜덤효과 구조를 확인하세요."
        ) from error
    for item in caught:
        _append_unique_warning(warning_messages, str(item.message))

    confidence_intervals = fitted.conf_int()
    coefficients = [
        ModelCoefficient(
            term=str(term),
            estimate=float(fitted.fe_params[term]),
            standard_error=float(fitted.bse_fe[term]),
            statistic=float(fitted.tvalues[term]),
            p_value=float(fitted.pvalues[term]),
            confidence_interval_lower=float(confidence_intervals.loc[term, 0]),
            confidence_interval_upper=float(confidence_intervals.loc[term, 1]),
        )
        for term in fitted.fe_params.index
    ]

    cov_re = np.asarray(fitted.cov_re, dtype=float)
    level3_intercept_variance = float(cov_re[0, 0])
    vc_names = list(getattr(fitted.model.exog_vc, "names", []))
    vc_values = [float(v) for v in np.asarray(fitted.vcomp, dtype=float)]
    variance_components = dict(zip(vc_names, vc_values, strict=False))
    level2_intercept_variance = float(variance_components.get("level2_intercept", 0.0))
    residual_variance = float(fitted.scale)
    total_variance = level3_intercept_variance + level2_intercept_variance + residual_variance
    level3_icc = level3_intercept_variance / total_variance if total_variance > 0 else None
    level2_icc = level2_intercept_variance / total_variance if total_variance > 0 else None

    fit_statistics: dict[str, Any] = {
        "log_likelihood": float(fitted.llf),
        "aic": _finite_float_or_none(fitted.aic),
        "bic": _finite_float_or_none(fitted.bic),
        "level3_intercept_variance": level3_intercept_variance,
        "level2_intercept_variance": level2_intercept_variance,
        "residual_variance": residual_variance,
        "level3_intraclass_correlation": level3_icc,
        "level2_intraclass_correlation": level2_icc,
        "combined_intraclass_correlation": (
            (level3_intercept_variance + level2_intercept_variance) / total_variance
            if total_variance > 0
            else None
        ),
        "random_intercept_variance": level3_intercept_variance + level2_intercept_variance,
        "intraclass_correlation": (
            (level3_intercept_variance + level2_intercept_variance) / total_variance
            if total_variance > 0
            else None
        ),
        "group_count": int(complete[level3_group].nunique()),
        "minimum_group_size": int(complete.groupby(level3_group, observed=True).size().min()),
        "maximum_group_size": int(complete.groupby(level3_group, observed=True).size().max()),
        "variance_partition": {
            "level1": residual_variance / total_variance if total_variance > 0 else None,
            "level2": level2_icc,
            "level3": level3_icc,
        },
        "level3_group_count": int(complete[level3_group].nunique()),
        "level2_group_count": int(working["__level2_nested__"].nunique()),
        "minimum_level3_size": int(complete.groupby(level3_group, observed=True).size().min()),
        "minimum_level2_size": int(
            working.groupby("__level2_nested__", observed=True).size().min()
        ),
        "variance_components": variance_components,
        "level3_random_effect_covariance_matrix": cov_re.tolist(),
    }
    if level3_slopes:
        fit_statistics["level3_random_slope_variances"] = {
            slope: float(cov_re[index + 1, index + 1]) for index, slope in enumerate(level3_slopes)
        }
    if level2_slopes:
        fit_statistics["level2_random_slope_variances"] = {
            slope: float(variance_components.get(f"level2_slope:{slope}", 0.0))
            for slope in level2_slopes
        }

    near_zero = {
        name: value
        for name, value in {
            "level3_intercept": level3_intercept_variance,
            "level2_intercept": level2_intercept_variance,
            **{
                f"level3_slope:{k}": v
                for k, v in fit_statistics.get("level3_random_slope_variances", {}).items()
            },
            **{
                f"level2_slope:{k}": v
                for k, v in fit_statistics.get("level2_random_slope_variances", {}).items()
            },
        }.items()
        if value <= max(1e-8, residual_variance * 1e-6)
    }
    if near_zero:
        _append_unique_warning(
            warning_messages,
            "3수준 랜덤효과 중 분산이 0에 가까운 항이 있습니다: " + ", ".join(near_zero),
        )
    if int(complete[level3_group].nunique()) < 5:
        _append_unique_warning(
            warning_messages, "Level 3 그룹 수가 5개 미만이어서 분산 추정이 불안정할 수 있습니다."
        )

    return RegressionResult(
        model_id=model_id,
        model_type="mixed_three_level",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(len(complete)),
        coefficients=coefficients,
        fit_statistics=fit_statistics,
        converged=bool(getattr(fitted, "converged", False)),
        standard_error_type="model_based",
        warnings=warning_messages,
        metadata={
            "level2_group": level2_group,
            "level3_group": level3_group,
            "nested": True,
            "level2_random_slope_variables": level2_slopes,
            "level3_random_slope_variables": level3_slopes,
            "reml": reml,
            "optimizer": method,
        },
        raw_result=fitted,
    )
