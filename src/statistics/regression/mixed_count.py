"""Random Intercept를 포함한 혼합 포아송 회귀분석."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import sparse
from scipy.stats import norm
from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.regression.design_matrix import prepare_regression_design_matrix


def fit_mixed_poisson_random_intercept(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    group_variable: str,
    model_id: str = "mixed_poisson_1",
    add_intercept: bool = True,
    optimizer: str = "BFGS",
    max_iterations: int = 200,
    fe_prior_sd: float = 2.0,
    variance_prior_sd: float = 1.0,
) -> RegressionResult:
    """변분 베이즈 방식으로 Random Intercept 혼합 포아송 모형을 적합한다."""
    if not group_variable.strip():
        raise ValueError("혼합 포아송 모형에는 그룹변수가 필요합니다.")
    if group_variable not in dataframe.columns:
        raise KeyError(f"데이터에 그룹변수가 없습니다: {group_variable}")
    if fe_prior_sd <= 0 or variance_prior_sd <= 0:
        raise ValueError("사전분포 표준편차는 0보다 커야 합니다.")

    independent_variables = list(dict.fromkeys(independent_variables))
    if group_variable in {dependent_variable, *independent_variables}:
        raise ValueError("그룹변수는 종속변수 또는 독립변수와 중복될 수 없습니다.")

    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=[],
        model_label="혼합 포아송",
    )
    groups = dataframe.loc[design.outcome.index, group_variable]
    valid_group = groups.notna()
    outcome = design.outcome.loc[valid_group].astype(float)
    predictors = design.predictors.loc[valid_group].astype(float)
    groups = groups.loc[valid_group].astype(str)

    if (outcome < 0).any() or not np.allclose(outcome, np.round(outcome)):
        raise ValueError("혼합 포아송 종속변수는 0 이상의 정수여야 합니다.")
    if outcome.nunique() <= 1:
        raise ValueError("혼합 포아송 종속변수가 상수입니다.")
    if groups.nunique() < 2:
        raise ValueError("혼합 포아송 모형에는 최소 2개 그룹이 필요합니다.")

    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")

    categories = pd.Categorical(groups)
    random_design = sparse.csr_matrix(
        (np.ones(len(groups)), (np.arange(len(groups)), categories.codes)),
        shape=(len(groups), len(categories.categories)),
    )
    fixed_names = [str(column) for column in predictors.columns]
    random_names = [str(value) for value in categories.categories]

    model = PoissonBayesMixedGLM(
        outcome.to_numpy(),
        predictors.to_numpy(),
        random_design,
        np.zeros(random_design.shape[1], dtype=int),
        fe_p=float(fe_prior_sd),
        vcp_p=float(variance_prior_sd),
        fep_names=fixed_names,
        vcp_names=[group_variable],
        vc_names=random_names,
    )
    fitted = model.fit_vb(
        fit_method=optimizer,
        minim_opts={"maxiter": int(max_iterations)},
    )

    coefficients: list[ModelCoefficient] = []
    for term, estimate, standard_error in zip(
        fixed_names, fitted.fe_mean, fitted.fe_sd, strict=True
    ):
        statistic = float(estimate / standard_error)
        p_value = float(2 * norm.sf(abs(statistic)))
        lower = float(estimate - 1.96 * standard_error)
        upper = float(estimate + 1.96 * standard_error)
        coefficients.append(
            ModelCoefficient(
                term=term,
                estimate=float(estimate),
                standard_error=float(standard_error),
                statistic=statistic,
                p_value=p_value,
                confidence_interval_lower=lower,
                confidence_interval_upper=upper,
                exponentiated_estimate=float(np.exp(estimate)),
            )
        )

    random_intercept_sd = float(np.exp(fitted.vcp_mean[0]))
    random_intercept_variance = random_intercept_sd**2
    optimizer_result = fitted.optim_retvals
    converged = bool(getattr(optimizer_result, "success", False))
    warnings: list[str] = []
    if not converged:
        warnings.append("혼합 포아송 모형의 변분 베이즈 최적화가 수렴하지 않았습니다.")
    if groups.value_counts().min() < 5:
        warnings.append(
            "일부 그룹의 사례 수가 5개 미만이어서 Random Intercept 추정이 불안정할 수 있습니다."
        )

    return RegressionResult(
        model_id=model_id,
        model_type="mixed_poisson_random_intercept",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=len(outcome),
        coefficients=coefficients,
        fit_statistics={
            "group_count": int(groups.nunique()),
            "outcome_mean": float(outcome.mean()),
            "outcome_variance": float(outcome.var(ddof=1)),
            "zero_count": int((outcome == 0).sum()),
            "random_intercept_sd": random_intercept_sd,
            "random_intercept_variance": random_intercept_variance,
            "variational_objective": float(getattr(optimizer_result, "fun", np.nan)),
        },
        converged=converged,
        standard_error_type="variational_bayes_posterior_sd",
        warnings=warnings,
        metadata={
            "group_variable": group_variable,
            "add_intercept": add_intercept,
            "optimizer": optimizer,
            "max_iterations": int(max_iterations),
            "estimation_method": "variational_bayes",
            "distribution": "poisson",
            "random_effects": dict(zip(random_names, map(float, fitted.vc_mean), strict=True)),
            **design.metadata,
            "design_matrix_columns": fixed_names,
        },
        raw_result=fitted,
    )


def fit_mixed_poisson_random_slope(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    group_variable: str,
    random_slope_variable: str,
    model_id: str = "mixed_poisson_1",
    add_intercept: bool = True,
    optimizer: str = "BFGS",
    max_iterations: int = 200,
    fe_prior_sd: float = 2.0,
    variance_prior_sd: float = 1.0,
) -> RegressionResult:
    """독립 Random Intercept와 Random Slope를 포함한 혼합 포아송 모형을 적합한다."""
    if not random_slope_variable.strip():
        raise ValueError("혼합 포아송 Random Slope 모형에는 random_slope_variable이 필요합니다.")
    if random_slope_variable not in independent_variables:
        raise ValueError("Random Slope 변수는 독립변수에 포함되어야 합니다.")
    if not group_variable.strip():
        raise ValueError("혼합 포아송 모형에는 그룹변수가 필요합니다.")
    if group_variable not in dataframe.columns:
        raise KeyError(f"데이터에 그룹변수가 없습니다: {group_variable}")
    if fe_prior_sd <= 0 or variance_prior_sd <= 0:
        raise ValueError("사전분포 표준편차는 0보다 커야 합니다.")

    independent_variables = list(dict.fromkeys(independent_variables))
    if group_variable in {dependent_variable, *independent_variables}:
        raise ValueError("그룹변수는 종속변수 또는 독립변수와 중복될 수 없습니다.")

    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=[],
        model_label="혼합 포아송 Random Slope",
    )
    groups = dataframe.loc[design.outcome.index, group_variable]
    valid_group = groups.notna()
    outcome = design.outcome.loc[valid_group].astype(float)
    predictors = design.predictors.loc[valid_group].astype(float)
    groups = groups.loc[valid_group].astype(str)

    if (outcome < 0).any() or not np.allclose(outcome, np.round(outcome)):
        raise ValueError("혼합 포아송 종속변수는 0 이상의 정수여야 합니다.")
    if outcome.nunique() <= 1:
        raise ValueError("혼합 포아송 종속변수가 상수입니다.")
    if groups.nunique() < 2:
        raise ValueError("혼합 포아송 모형에는 최소 2개 그룹이 필요합니다.")

    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")

    categories = pd.Categorical(groups)
    row_index = np.arange(len(groups))
    column_index = categories.codes
    group_count = len(categories.categories)
    intercept_design = sparse.csr_matrix(
        (np.ones(len(groups)), (row_index, column_index)),
        shape=(len(groups), group_count),
    )
    slope_values = predictors[random_slope_variable].to_numpy(dtype=float)
    slope_design = sparse.csr_matrix(
        (slope_values, (row_index, column_index)),
        shape=(len(groups), group_count),
    )
    random_design = sparse.hstack([intercept_design, slope_design], format="csr")
    ident = np.concatenate([np.zeros(group_count, dtype=int), np.ones(group_count, dtype=int)])
    fixed_names = [str(column) for column in predictors.columns]
    group_names = [str(value) for value in categories.categories]
    random_names = [f"{name}:intercept" for name in group_names] + [
        f"{name}:{random_slope_variable}" for name in group_names
    ]

    model = PoissonBayesMixedGLM(
        outcome.to_numpy(),
        predictors.to_numpy(),
        random_design,
        ident,
        fe_p=float(fe_prior_sd),
        vcp_p=float(variance_prior_sd),
        fep_names=fixed_names,
        vcp_names=[group_variable, random_slope_variable],
        vc_names=random_names,
    )
    fitted = model.fit_vb(
        fit_method=optimizer,
        minim_opts={"maxiter": int(max_iterations)},
    )

    coefficients: list[ModelCoefficient] = []
    for term, estimate, standard_error in zip(
        fixed_names, fitted.fe_mean, fitted.fe_sd, strict=True
    ):
        statistic = float(estimate / standard_error)
        p_value = float(2 * norm.sf(abs(statistic)))
        lower = float(estimate - 1.96 * standard_error)
        upper = float(estimate + 1.96 * standard_error)
        coefficients.append(
            ModelCoefficient(
                term=term,
                estimate=float(estimate),
                standard_error=float(standard_error),
                statistic=statistic,
                p_value=p_value,
                confidence_interval_lower=lower,
                confidence_interval_upper=upper,
                exponentiated_estimate=float(np.exp(estimate)),
            )
        )

    random_intercept_sd = float(np.exp(fitted.vcp_mean[0]))
    random_slope_sd = float(np.exp(fitted.vcp_mean[1]))
    optimizer_result = fitted.optim_retvals
    converged = bool(getattr(optimizer_result, "success", False))
    warnings: list[str] = []
    if not converged:
        warnings.append("혼합 포아송 모형의 변분 베이즈 최적화가 수렴하지 않았습니다.")
    if groups.value_counts().min() < 5:
        warnings.append(
            "일부 그룹의 사례 수가 5개 미만이어서 Random Slope 추정이 불안정할 수 있습니다."
        )

    random_effect_values = dict(zip(random_names, map(float, fitted.vc_mean), strict=True))
    return RegressionResult(
        model_id=model_id,
        model_type="mixed_poisson_random_slope",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=len(outcome),
        coefficients=coefficients,
        fit_statistics={
            "group_count": int(groups.nunique()),
            "outcome_mean": float(outcome.mean()),
            "outcome_variance": float(outcome.var(ddof=1)),
            "zero_count": int((outcome == 0).sum()),
            "random_intercept_sd": random_intercept_sd,
            "random_intercept_variance": random_intercept_sd**2,
            "random_slope_sd": random_slope_sd,
            "random_slope_variance": random_slope_sd**2,
            "variational_objective": float(getattr(optimizer_result, "fun", np.nan)),
        },
        converged=converged,
        standard_error_type="variational_bayes_posterior_sd",
        warnings=warnings,
        metadata={
            "group_variable": group_variable,
            "random_slope_variable": random_slope_variable,
            "random_effect_covariance": "diagonal",
            "add_intercept": add_intercept,
            "optimizer": optimizer,
            "max_iterations": int(max_iterations),
            "estimation_method": "variational_bayes",
            "distribution": "poisson",
            "random_effects": random_effect_values,
            "random_intercepts": {
                name: random_effect_values[f"{name}:intercept"] for name in group_names
            },
            "random_slopes": {
                name: random_effect_values[f"{name}:{random_slope_variable}"]
                for name in group_names
            },
            **design.metadata,
            "design_matrix_columns": fixed_names,
        },
        raw_result=fitted,
    )


def fit_mixed_poisson_three_level(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    level2_group: str,
    level3_group: str,
    model_id: str = "mixed_poisson_1",
    add_intercept: bool = True,
    optimizer: str = "BFGS",
    max_iterations: int = 300,
    fe_prior_sd: float = 2.0,
    variance_prior_sd: float = 1.0,
) -> RegressionResult:
    """중첩된 Level 2·Level 3 Random Intercept 혼합 포아송 모형을 적합한다."""
    if not level2_group.strip() or not level3_group.strip():
        raise ValueError("3수준 혼합 포아송에는 level2_group과 level3_group이 필요합니다.")
    if level2_group == level3_group:
        raise ValueError("level2_group과 level3_group은 서로 달라야 합니다.")
    for group_variable in (level2_group, level3_group):
        if group_variable not in dataframe.columns:
            raise KeyError(f"데이터에 그룹변수가 없습니다: {group_variable}")
    if fe_prior_sd <= 0 or variance_prior_sd <= 0:
        raise ValueError("사전분포 표준편차는 0보다 커야 합니다.")

    independent_variables = list(dict.fromkeys(independent_variables))
    reserved = {dependent_variable, *independent_variables}
    if level2_group in reserved or level3_group in reserved:
        raise ValueError("그룹변수는 종속변수 또는 독립변수와 중복될 수 없습니다.")

    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=[],
        model_label="3수준 혼합 포아송",
    )
    groups = dataframe.loc[design.outcome.index, [level2_group, level3_group]]
    valid_group = groups.notna().all(axis=1)
    outcome = design.outcome.loc[valid_group].astype(float)
    predictors = design.predictors.loc[valid_group].astype(float)
    groups = groups.loc[valid_group].astype(str)

    if (outcome < 0).any() or not np.allclose(outcome, np.round(outcome)):
        raise ValueError("혼합 포아송 종속변수는 0 이상의 정수여야 합니다.")
    if outcome.nunique() <= 1:
        raise ValueError("혼합 포아송 종속변수가 상수입니다.")
    if groups[level2_group].nunique() < 2 or groups[level3_group].nunique() < 2:
        raise ValueError("3수준 혼합 포아송에는 각 상위 수준별 최소 2개 그룹이 필요합니다.")
    nesting_counts = groups.groupby(level2_group, observed=True)[level3_group].nunique()
    if (nesting_counts > 1).any():
        raise ValueError("각 Level 2 그룹은 하나의 Level 3 그룹에만 중첩되어야 합니다.")

    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")

    row_index = np.arange(len(groups))
    level2_categories = pd.Categorical(groups[level2_group])
    level3_categories = pd.Categorical(groups[level3_group])
    level2_count = len(level2_categories.categories)
    level3_count = len(level3_categories.categories)
    level2_design = sparse.csr_matrix(
        (np.ones(len(groups)), (row_index, level2_categories.codes)),
        shape=(len(groups), level2_count),
    )
    level3_design = sparse.csr_matrix(
        (np.ones(len(groups)), (row_index, level3_categories.codes)),
        shape=(len(groups), level3_count),
    )
    random_design = sparse.hstack([level2_design, level3_design], format="csr")
    ident = np.concatenate([np.zeros(level2_count, dtype=int), np.ones(level3_count, dtype=int)])
    fixed_names = [str(column) for column in predictors.columns]
    level2_names = [str(value) for value in level2_categories.categories]
    level3_names = [str(value) for value in level3_categories.categories]
    random_names = [f"{level2_group}:{name}" for name in level2_names] + [
        f"{level3_group}:{name}" for name in level3_names
    ]

    model = PoissonBayesMixedGLM(
        outcome.to_numpy(),
        predictors.to_numpy(),
        random_design,
        ident,
        fe_p=float(fe_prior_sd),
        vcp_p=float(variance_prior_sd),
        fep_names=fixed_names,
        vcp_names=[level2_group, level3_group],
        vc_names=random_names,
    )
    fitted = model.fit_vb(
        fit_method=optimizer,
        minim_opts={"maxiter": int(max_iterations)},
    )

    coefficients: list[ModelCoefficient] = []
    for term, estimate, standard_error in zip(
        fixed_names, fitted.fe_mean, fitted.fe_sd, strict=True
    ):
        statistic = float(estimate / standard_error)
        p_value = float(2 * norm.sf(abs(statistic)))
        lower = float(estimate - 1.96 * standard_error)
        upper = float(estimate + 1.96 * standard_error)
        coefficients.append(
            ModelCoefficient(
                term=term,
                estimate=float(estimate),
                standard_error=float(standard_error),
                statistic=statistic,
                p_value=p_value,
                confidence_interval_lower=lower,
                confidence_interval_upper=upper,
                exponentiated_estimate=float(np.exp(estimate)),
            )
        )

    level2_sd = float(np.exp(fitted.vcp_mean[0]))
    level3_sd = float(np.exp(fitted.vcp_mean[1]))
    level2_variance = level2_sd**2
    level3_variance = level3_sd**2
    total_random_variance = level2_variance + level3_variance
    level2_vpc = level2_variance / total_random_variance
    level3_vpc = level3_variance / total_random_variance
    optimizer_result = fitted.optim_retvals
    converged = bool(getattr(optimizer_result, "success", False))
    warnings: list[str] = []
    if not converged:
        warnings.append("3수준 혼합 포아송의 변분 베이즈 최적화가 수렴하지 않았습니다.")
    if groups[level2_group].value_counts().min() < 5:
        warnings.append("일부 Level 2 그룹의 사례 수가 5개 미만이어서 추정이 불안정할 수 있습니다.")
    level2_per_level3 = groups.drop_duplicates().groupby(level3_group, observed=True).size()
    if level2_per_level3.min() < 2:
        warnings.append("일부 Level 3 그룹에 중첩된 Level 2 그룹이 2개 미만입니다.")

    vc_mean = list(map(float, fitted.vc_mean))
    return RegressionResult(
        model_id=model_id,
        model_type="mixed_poisson_three_level",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=len(outcome),
        coefficients=coefficients,
        fit_statistics={
            "level2_group_count": level2_count,
            "level3_group_count": level3_count,
            "outcome_mean": float(outcome.mean()),
            "outcome_variance": float(outcome.var(ddof=1)),
            "zero_count": int((outcome == 0).sum()),
            "level2_random_intercept_sd": level2_sd,
            "level2_random_intercept_variance": level2_variance,
            "level3_random_intercept_sd": level3_sd,
            "level3_random_intercept_variance": level3_variance,
            "level2_vpc": float(level2_vpc),
            "level3_vpc": float(level3_vpc),
            "variational_objective": float(getattr(optimizer_result, "fun", np.nan)),
        },
        converged=converged,
        standard_error_type="variational_bayes_posterior_sd",
        warnings=warnings,
        metadata={
            "level2_group": level2_group,
            "level3_group": level3_group,
            "nested_structure": True,
            "add_intercept": add_intercept,
            "optimizer": optimizer,
            "max_iterations": int(max_iterations),
            "estimation_method": "variational_bayes",
            "distribution": "poisson",
            "level2_random_effects": dict(zip(level2_names, vc_mean[:level2_count], strict=True)),
            "level3_random_effects": dict(zip(level3_names, vc_mean[level2_count:], strict=True)),
            **design.metadata,
            "design_matrix_columns": fixed_names,
        },
        raw_result=fitted,
    )
