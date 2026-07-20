"""Negative-binomial mixed-effects regression models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
from scipy.stats import norm

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.regression.design_matrix import prepare_regression_design_matrix


def _negative_binomial_loglike(
    y: np.ndarray,
    eta: np.ndarray,
    alpha: float,
) -> np.ndarray:
    mu = np.exp(np.clip(eta, -30, 30))
    size = 1.0 / alpha
    return (
        gammaln(y + size)
        - gammaln(size)
        - gammaln(y + 1.0)
        + size * (np.log(size) - np.log(size + mu))
        + y * (np.log(mu) - np.log(size + mu))
    )


def _hessian_covariance(fitted: object, parameter_count: int) -> np.ndarray:
    covariance = np.asarray(fitted.hess_inv, dtype=float)
    if covariance.shape != (parameter_count, parameter_count):
        return np.full((parameter_count, parameter_count), np.nan)
    return covariance


def _fixed_effect_coefficients(
    names: list[str],
    estimates: np.ndarray,
    covariance: np.ndarray,
) -> list[ModelCoefficient]:
    standard_errors = np.sqrt(np.clip(np.diag(covariance)[: len(estimates)], 0, np.inf))
    coefficients: list[ModelCoefficient] = []
    for term, estimate, standard_error in zip(names, estimates, standard_errors, strict=True):
        statistic = float(estimate / standard_error) if standard_error > 0 else np.nan
        p_value = float(2 * norm.sf(abs(statistic))) if np.isfinite(statistic) else np.nan
        coefficients.append(
            ModelCoefficient(
                term=term,
                estimate=float(estimate),
                standard_error=float(standard_error),
                statistic=statistic,
                p_value=p_value,
                confidence_interval_lower=float(estimate - 1.96 * standard_error),
                confidence_interval_upper=float(estimate + 1.96 * standard_error),
                exponentiated_estimate=float(np.exp(estimate)),
            )
        )
    return coefficients


def _prepare_count_design(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    model_label: str,
) -> tuple[object, pd.Series, pd.DataFrame]:
    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=[],
        model_label=model_label,
    )
    outcome = design.outcome.astype(float)
    predictors = design.predictors.astype(float)
    if (outcome < 0).any() or not np.allclose(outcome, np.round(outcome)):
        raise ValueError("mixed negative binomial outcome must be a nonnegative integer.")
    if outcome.nunique() <= 1:
        raise ValueError("mixed negative binomial outcome is constant.")
    return design, outcome, predictors


def _conditional_random_intercepts(
    *,
    y: np.ndarray,
    x: np.ndarray,
    beta: np.ndarray,
    alpha: float,
    sigma: float,
    group_indices: list[np.ndarray],
    group_names: list[str],
) -> dict[str, float]:
    if sigma <= np.finfo(float).eps:
        return {name: 0.0 for name in group_names}

    estimates: dict[str, float] = {}
    variance = sigma**2
    for name, idx in zip(group_names, group_indices, strict=True):
        eta = x[idx] @ beta
        yi = y[idx]

        def objective(
            value: np.ndarray,
            *,
            current_y: np.ndarray = yi,
            current_eta: np.ndarray = eta,
        ) -> float:
            random_intercept = float(value[0])
            loglike = _negative_binomial_loglike(
                current_y, current_eta + random_intercept, alpha
            ).sum()
            penalty = 0.5 * random_intercept**2 / variance
            return float(-(loglike - penalty))

        fitted = minimize(objective, np.zeros(1), method="BFGS")
        estimates[name] = float(fitted.x[0]) if fitted.success else 0.0
    return estimates


def _conditional_random_intercept_slopes(
    *,
    y: np.ndarray,
    x: np.ndarray,
    slope_values: np.ndarray,
    beta: np.ndarray,
    alpha: float,
    intercept_sd: float,
    slope_sd: float,
    group_indices: list[np.ndarray],
    group_names: list[str],
) -> tuple[dict[str, float], dict[str, float]]:
    intercepts: dict[str, float] = {}
    slopes: dict[str, float] = {}
    intercept_variance = max(intercept_sd**2, np.finfo(float).eps)
    slope_variance = max(slope_sd**2, np.finfo(float).eps)

    for name, idx in zip(group_names, group_indices, strict=True):
        eta = x[idx] @ beta
        zi = slope_values[idx]
        yi = y[idx]

        def objective(
            values: np.ndarray,
            *,
            current_y: np.ndarray = yi,
            current_eta: np.ndarray = eta,
            current_z: np.ndarray = zi,
        ) -> float:
            random_intercept = float(values[0])
            random_slope = float(values[1])
            loglike = _negative_binomial_loglike(
                current_y,
                current_eta + random_intercept + random_slope * current_z,
                alpha,
            ).sum()
            penalty = (
                0.5 * random_intercept**2 / intercept_variance
                + 0.5 * random_slope**2 / slope_variance
            )
            return float(-(loglike - penalty))

        fitted = minimize(objective, np.zeros(2), method="BFGS")
        values = fitted.x if fitted.success else np.zeros(2)
        intercepts[name] = float(values[0])
        slopes[name] = float(values[1])

    return intercepts, slopes


def fit_mixed_negative_binomial_random_intercept(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    group_variable: str,
    model_id: str = "mixed_negative_binomial_1",
    add_intercept: bool = True,
    optimizer: str = "BFGS",
    max_iterations: int = 300,
    quadrature_points: int = 15,
) -> RegressionResult:
    """Fit an NB2 GLMM with a random intercept."""
    if not group_variable.strip():
        raise ValueError("mixed negative binomial model requires group_variable.")
    if group_variable not in dataframe.columns:
        raise KeyError(f"group variable is missing from dataframe: {group_variable}")
    if quadrature_points < 5:
        raise ValueError("quadrature_points must be at least 5.")

    independent_variables = list(dict.fromkeys(independent_variables))
    if group_variable in {dependent_variable, *independent_variables}:
        raise ValueError("group_variable cannot duplicate the outcome or predictors.")

    design, outcome, predictors = _prepare_count_design(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        model_label="mixed negative binomial",
    )
    groups = dataframe.loc[design.outcome.index, group_variable]
    valid_group = groups.notna()
    outcome = outcome.loc[valid_group]
    predictors = predictors.loc[valid_group]
    groups = groups.loc[valid_group].astype(str)

    if groups.nunique() < 2:
        raise ValueError("mixed negative binomial model requires at least 2 groups.")
    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")

    y = outcome.to_numpy(dtype=float)
    x = predictors.to_numpy(dtype=float)
    group_categories = pd.Categorical(groups)
    group_codes = group_categories.codes
    group_indices = [np.flatnonzero(group_codes == code) for code in np.unique(group_codes)]
    group_names = [str(value) for value in group_categories.categories]
    nodes, weights = hermgauss(quadrature_points)
    log_weights = np.log(weights) - 0.5 * np.log(np.pi)

    def objective(params: np.ndarray) -> float:
        beta = params[:-2]
        sigma = float(np.exp(np.clip(params[-2], -20, 5)))
        alpha = float(np.exp(np.clip(params[-1], -20, 5)))
        total = 0.0
        random_values = np.sqrt(2.0) * sigma * nodes
        for idx in group_indices:
            eta = x[idx] @ beta
            yi = y[idx]
            node_ll = [
                _negative_binomial_loglike(yi, eta + random_intercept, alpha).sum()
                for random_intercept in random_values
            ]
            total += logsumexp(log_weights + np.asarray(node_ll))
        return float(-total)

    poisson = sm.GLM(y, x, family=sm.families.Poisson()).fit()
    initial = np.concatenate([np.asarray(poisson.params), [np.log(0.3), np.log(0.3)]])
    fitted = minimize(
        objective,
        initial,
        method=optimizer,
        options={"maxiter": int(max_iterations)},
    )

    params = np.asarray(fitted.x, dtype=float)
    beta = params[:-2]
    covariance = _hessian_covariance(fitted, len(params))
    fixed_names = [str(column) for column in predictors.columns]
    coefficients = _fixed_effect_coefficients(fixed_names, beta, covariance)
    random_intercept_sd = float(np.exp(params[-2]))
    alpha = float(np.exp(params[-1]))
    random_effects = _conditional_random_intercepts(
        y=y,
        x=x,
        beta=beta,
        alpha=alpha,
        sigma=random_intercept_sd,
        group_indices=group_indices,
        group_names=group_names,
    )
    random_intercept_values = np.asarray(
        [random_effects[str(group)] for group in groups], dtype=float
    )
    predicted_mean = np.exp(np.clip(x @ beta + random_intercept_values, -30, 30))

    warnings: list[str] = []
    if not fitted.success:
        warnings.append("mixed negative binomial optimization did not converge.")
    if groups.value_counts().min() < 5:
        warnings.append("some groups have fewer than 5 observations; random effects may be unstable.")

    return RegressionResult(
        model_id=model_id,
        model_type="mixed_negative_binomial_random_intercept",
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
            "dispersion_alpha": alpha,
            "log_likelihood": float(-fitted.fun),
            "aic": float(2 * len(params) + 2 * fitted.fun),
        },
        converged=bool(fitted.success),
        standard_error_type="maximum_likelihood_hessian",
        warnings=warnings,
        metadata={
            "group_variable": group_variable,
            "add_intercept": add_intercept,
            "optimizer": optimizer,
            "max_iterations": int(max_iterations),
            "quadrature_points": int(quadrature_points),
            "estimation_method": "adaptive_gauss_hermite_quadrature",
            "distribution": "negative_binomial_2",
            "random_effects": random_effects,
            "diagnostics": {
                "endog": y.tolist(),
                "predicted_mean": predicted_mean.tolist(),
                "exog": x.tolist(),
                "exog_names": fixed_names,
                "row_labels": outcome.index.tolist(),
            },
            **design.metadata,
            "design_matrix_columns": fixed_names,
        },
        raw_result=fitted,
    )


def fit_mixed_negative_binomial_random_slope(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    group_variable: str,
    random_slope_variable: str,
    model_id: str = "mixed_negative_binomial_1",
    add_intercept: bool = True,
    optimizer: str = "BFGS",
    max_iterations: int = 300,
    quadrature_points: int = 9,
) -> RegressionResult:
    """Fit an NB2 GLMM with independent random intercept and random slope."""
    if not random_slope_variable.strip():
        raise ValueError("mixed negative binomial random slope requires random_slope_variable.")
    if random_slope_variable not in independent_variables:
        raise ValueError("Random slope variable must be included in independent_variables.")
    if not group_variable.strip():
        raise ValueError("mixed negative binomial model requires group_variable.")
    if group_variable not in dataframe.columns:
        raise KeyError(f"group variable is missing from dataframe: {group_variable}")
    if quadrature_points < 5:
        raise ValueError("quadrature_points must be at least 5.")

    independent_variables = list(dict.fromkeys(independent_variables))
    if group_variable in {dependent_variable, *independent_variables}:
        raise ValueError("group_variable cannot duplicate the outcome or predictors.")

    design, outcome, predictors = _prepare_count_design(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        model_label="mixed negative binomial random slope",
    )
    groups = dataframe.loc[design.outcome.index, group_variable]
    valid_group = groups.notna()
    outcome = outcome.loc[valid_group]
    predictors = predictors.loc[valid_group]
    groups = groups.loc[valid_group].astype(str)

    if groups.nunique() < 2:
        raise ValueError("mixed negative binomial model requires at least 2 groups.")
    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")

    y = outcome.to_numpy(dtype=float)
    x = predictors.to_numpy(dtype=float)
    z = predictors[random_slope_variable].to_numpy(dtype=float)
    group_categories = pd.Categorical(groups)
    group_codes = group_categories.codes
    group_indices = [np.flatnonzero(group_codes == code) for code in np.unique(group_codes)]
    group_names = [str(value) for value in group_categories.categories]
    nodes, weights = hermgauss(quadrature_points)
    log_weights = np.log(weights) - 0.5 * np.log(np.pi)
    grid_weights = (log_weights[:, None] + log_weights[None, :]).ravel()

    def objective(params: np.ndarray) -> float:
        beta = params[:-3]
        intercept_sd = float(np.exp(np.clip(params[-3], -20, 5)))
        slope_sd = float(np.exp(np.clip(params[-2], -20, 5)))
        alpha = float(np.exp(np.clip(params[-1], -20, 5)))
        intercept_values = np.sqrt(2.0) * intercept_sd * nodes
        slope_values = np.sqrt(2.0) * slope_sd * nodes
        total = 0.0
        for idx in group_indices:
            eta = x[idx] @ beta
            zi = z[idx]
            yi = y[idx]
            node_ll = []
            for random_intercept in intercept_values:
                for random_slope in slope_values:
                    node_ll.append(
                        _negative_binomial_loglike(
                            yi,
                            eta + random_intercept + random_slope * zi,
                            alpha,
                        ).sum()
                    )
            total += logsumexp(grid_weights + np.asarray(node_ll))
        return float(-total)

    poisson = sm.GLM(y, x, family=sm.families.Poisson()).fit()
    initial = np.concatenate(
        [np.asarray(poisson.params), [np.log(0.25), np.log(0.15), np.log(0.3)]]
    )
    fitted = minimize(
        objective,
        initial,
        method=optimizer,
        options={"maxiter": int(max_iterations)},
    )

    params = np.asarray(fitted.x, dtype=float)
    beta = params[:-3]
    covariance = _hessian_covariance(fitted, len(params))
    fixed_names = [str(column) for column in predictors.columns]
    coefficients = _fixed_effect_coefficients(fixed_names, beta, covariance)
    random_intercept_sd = float(np.exp(params[-3]))
    random_slope_sd = float(np.exp(params[-2]))
    alpha = float(np.exp(params[-1]))
    random_intercepts, random_slopes = _conditional_random_intercept_slopes(
        y=y,
        x=x,
        slope_values=z,
        beta=beta,
        alpha=alpha,
        intercept_sd=random_intercept_sd,
        slope_sd=random_slope_sd,
        group_indices=group_indices,
        group_names=group_names,
    )
    random_intercept_values = np.asarray(
        [random_intercepts[str(group)] for group in groups], dtype=float
    )
    random_slope_values = np.asarray([random_slopes[str(group)] for group in groups], dtype=float)
    predicted_mean = np.exp(
        np.clip(x @ beta + random_intercept_values + random_slope_values * z, -30, 30)
    )

    warnings: list[str] = []
    if not fitted.success:
        warnings.append("mixed negative binomial random slope optimization did not converge.")
    if groups.value_counts().min() < 5:
        warnings.append("some groups have fewer than 5 observations; random effects may be unstable.")

    return RegressionResult(
        model_id=model_id,
        model_type="mixed_negative_binomial_random_slope",
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
            "dispersion_alpha": alpha,
            "log_likelihood": float(-fitted.fun),
            "aic": float(2 * len(params) + 2 * fitted.fun),
        },
        converged=bool(fitted.success),
        standard_error_type="maximum_likelihood_hessian",
        warnings=warnings,
        metadata={
            "group_variable": group_variable,
            "random_slope_variable": random_slope_variable,
            "random_effect_covariance": "diagonal",
            "add_intercept": add_intercept,
            "optimizer": optimizer,
            "max_iterations": int(max_iterations),
            "quadrature_points": int(quadrature_points),
            "estimation_method": "adaptive_gauss_hermite_quadrature",
            "distribution": "negative_binomial_2",
            "random_intercepts": random_intercepts,
            "random_slopes": random_slopes,
            "random_effects": random_intercepts,
            "diagnostics": {
                "endog": y.tolist(),
                "predicted_mean": predicted_mean.tolist(),
                "exog": x.tolist(),
                "exog_names": fixed_names,
                "row_labels": outcome.index.tolist(),
            },
            **design.metadata,
            "design_matrix_columns": fixed_names,
        },
        raw_result=fitted,
    )


def fit_mixed_negative_binomial_three_level(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    level2_group: str,
    level3_group: str,
    model_id: str = "mixed_negative_binomial_1",
    add_intercept: bool = True,
    optimizer: str = "BFGS",
    max_iterations: int = 300,
    quadrature_points: int = 7,
) -> RegressionResult:
    """Fit a nested three-level NB2 GLMM with random intercepts at levels 2 and 3."""
    if not level2_group.strip() or not level3_group.strip():
        raise ValueError("three-level mixed negative binomial requires level2_group and level3_group.")
    if level2_group == level3_group:
        raise ValueError("level2_group and level3_group must differ.")
    for group_variable in (level2_group, level3_group):
        if group_variable not in dataframe.columns:
            raise KeyError(f"group variable is missing from dataframe: {group_variable}")
    if quadrature_points < 5:
        raise ValueError("quadrature_points must be at least 5.")

    independent_variables = list(dict.fromkeys(independent_variables))
    reserved = {dependent_variable, *independent_variables}
    if level2_group in reserved or level3_group in reserved:
        raise ValueError("group variables cannot duplicate the outcome or predictors.")

    design, outcome, predictors = _prepare_count_design(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        model_label="three-level mixed negative binomial",
    )
    groups = dataframe.loc[design.outcome.index, [level2_group, level3_group]]
    valid_group = groups.notna().all(axis=1)
    outcome = outcome.loc[valid_group]
    predictors = predictors.loc[valid_group]
    groups = groups.loc[valid_group].astype(str)

    if groups[level2_group].nunique() < 2 or groups[level3_group].nunique() < 2:
        raise ValueError("three-level mixed negative binomial requires at least 2 groups per level.")
    nesting_counts = groups.groupby(level2_group, observed=True)[level3_group].nunique()
    if (nesting_counts > 1).any():
        raise ValueError("level2 groups must be nested in exactly one level3 group.")
    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")

    y = outcome.to_numpy(dtype=float)
    x = predictors.to_numpy(dtype=float)
    level2_categories = pd.Categorical(groups[level2_group])
    level3_categories = pd.Categorical(groups[level3_group])
    level2_codes = level2_categories.codes
    level3_codes = level3_categories.codes
    level2_group_indices = {
        int(code): np.flatnonzero(level2_codes == code) for code in np.unique(level2_codes)
    }
    level2_to_level3 = {
        code: int(level3_codes[idx[0]]) for code, idx in level2_group_indices.items()
    }
    level3_to_level2: dict[int, list[int]] = {}
    for level2_code, level3_code in level2_to_level3.items():
        level3_to_level2.setdefault(level3_code, []).append(level2_code)

    nodes, weights = hermgauss(quadrature_points)
    log_weights = np.log(weights) - 0.5 * np.log(np.pi)

    def objective(params: np.ndarray) -> float:
        beta = params[:-3]
        level2_sd = float(np.exp(np.clip(params[-3], -20, 5)))
        level3_sd = float(np.exp(np.clip(params[-2], -20, 5)))
        alpha = float(np.exp(np.clip(params[-1], -20, 5)))
        level2_values = np.sqrt(2.0) * level2_sd * nodes
        level3_values = np.sqrt(2.0) * level3_sd * nodes
        total = 0.0
        for level2_codes_for_level3 in level3_to_level2.values():
            level3_node_ll = []
            for level3_random_intercept in level3_values:
                nested_total = 0.0
                for level2_code in level2_codes_for_level3:
                    idx = level2_group_indices[level2_code]
                    eta = x[idx] @ beta
                    yi = y[idx]
                    level2_node_ll = [
                        _negative_binomial_loglike(
                            yi,
                            eta + level2_random_intercept + level3_random_intercept,
                            alpha,
                        ).sum()
                        for level2_random_intercept in level2_values
                    ]
                    nested_total += logsumexp(log_weights + np.asarray(level2_node_ll))
                level3_node_ll.append(nested_total)
            total += logsumexp(log_weights + np.asarray(level3_node_ll))
        return float(-total)

    poisson = sm.GLM(y, x, family=sm.families.Poisson()).fit()
    initial = np.concatenate(
        [np.asarray(poisson.params), [np.log(0.25), np.log(0.25), np.log(0.3)]]
    )
    fitted = minimize(
        objective,
        initial,
        method=optimizer,
        options={"maxiter": int(max_iterations)},
    )

    params = np.asarray(fitted.x, dtype=float)
    beta = params[:-3]
    covariance = _hessian_covariance(fitted, len(params))
    fixed_names = [str(column) for column in predictors.columns]
    coefficients = _fixed_effect_coefficients(fixed_names, beta, covariance)
    level2_sd = float(np.exp(params[-3]))
    level3_sd = float(np.exp(params[-2]))
    level2_variance = level2_sd**2
    level3_variance = level3_sd**2
    total_random_variance = level2_variance + level3_variance
    level2_vpc = level2_variance / total_random_variance
    level3_vpc = level3_variance / total_random_variance
    alpha = float(np.exp(params[-1]))

    level2_names = [str(value) for value in level2_categories.categories]
    level3_names = [str(value) for value in level3_categories.categories]
    level2_indices = [level2_group_indices[int(code)] for code in np.unique(level2_codes)]
    level2_random_effects = _conditional_random_intercepts(
        y=y,
        x=x,
        beta=beta,
        alpha=alpha,
        sigma=level2_sd,
        group_indices=level2_indices,
        group_names=level2_names,
    )
    level3_random_effects = {name: 0.0 for name in level3_names}
    level2_random_values = np.asarray(
        [level2_random_effects[str(group)] for group in groups[level2_group]], dtype=float
    )
    level3_random_values = np.asarray(
        [level3_random_effects[str(group)] for group in groups[level3_group]], dtype=float
    )
    predicted_mean = np.exp(np.clip(x @ beta + level2_random_values + level3_random_values, -30, 30))

    warnings: list[str] = []
    if not fitted.success:
        warnings.append("three-level mixed negative binomial optimization did not converge.")
    if groups[level2_group].value_counts().min() < 5:
        warnings.append("some level2 groups have fewer than 5 observations; estimates may be unstable.")
    level2_per_level3 = groups.drop_duplicates().groupby(level3_group, observed=True).size()
    if level2_per_level3.min() < 2:
        warnings.append("some level3 groups contain fewer than 2 nested level2 groups.")

    return RegressionResult(
        model_id=model_id,
        model_type="mixed_negative_binomial_three_level",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=len(outcome),
        coefficients=coefficients,
        fit_statistics={
            "level2_group_count": len(level2_names),
            "level3_group_count": len(level3_names),
            "outcome_mean": float(outcome.mean()),
            "outcome_variance": float(outcome.var(ddof=1)),
            "zero_count": int((outcome == 0).sum()),
            "level2_random_intercept_sd": level2_sd,
            "level2_random_intercept_variance": level2_variance,
            "level3_random_intercept_sd": level3_sd,
            "level3_random_intercept_variance": level3_variance,
            "level2_vpc": float(level2_vpc),
            "level3_vpc": float(level3_vpc),
            "dispersion_alpha": alpha,
            "log_likelihood": float(-fitted.fun),
            "aic": float(2 * len(params) + 2 * fitted.fun),
        },
        converged=bool(fitted.success),
        standard_error_type="maximum_likelihood_hessian",
        warnings=warnings,
        metadata={
            "level2_group": level2_group,
            "level3_group": level3_group,
            "nested_structure": True,
            "add_intercept": add_intercept,
            "optimizer": optimizer,
            "max_iterations": int(max_iterations),
            "quadrature_points": int(quadrature_points),
            "estimation_method": "adaptive_gauss_hermite_quadrature",
            "distribution": "negative_binomial_2",
            "level2_random_effects": level2_random_effects,
            "level3_random_effects": level3_random_effects,
            "random_effects": level2_random_effects,
            "diagnostics": {
                "endog": y.tolist(),
                "predicted_mean": predicted_mean.tolist(),
                "exog": x.tolist(),
                "exog_names": fixed_names,
                "row_labels": outcome.index.tolist(),
            },
            **design.metadata,
            "design_matrix_columns": fixed_names,
        },
        raw_result=fitted,
    )
