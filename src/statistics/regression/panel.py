"""Panel fixed-effects regression."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.statistics.regression.base import (
    ModelCoefficient,
    RegressionResult,
    validate_model_variables,
)


def _ordered_categories(series: pd.Series) -> list[Any]:
    values = series.dropna().drop_duplicates().tolist()
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda value: str(value))


def _residualize_against_fixed_effects(values: np.ndarray, fixed_effect_matrix: np.ndarray) -> np.ndarray:
    if fixed_effect_matrix.size == 0:
        return values.astype(float)
    fitted = fixed_effect_matrix @ np.linalg.lstsq(fixed_effect_matrix, values, rcond=None)[0]
    return values - fitted


def _fixed_effect_matrix(work: pd.DataFrame, entity_variable: str, time_variable: str | None) -> np.ndarray:
    parts = [np.ones((len(work), 1), dtype=float)]
    for variable in [entity_variable, time_variable]:
        if variable is None:
            continue
        categories = _ordered_categories(work[variable])
        if len(categories) <= 1:
            continue
        dummies = pd.get_dummies(
            pd.Categorical(work[variable], categories=categories),
            drop_first=True,
            dtype=float,
        )
        parts.append(dummies.to_numpy(dtype=float))
    return np.column_stack(parts)


def fit_panel_fixed_effects(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    entity_variable: str,
    time_variable: str | None = None,
    model_id: str = "panel_fixed_effects_1",
    covariance_type: str = "cluster_entity",
) -> RegressionResult:
    """Fit a within-transformed fixed-effects panel regression."""
    if covariance_type not in {"nonrobust", "HC3", "cluster_entity"}:
        raise ValueError("Panel fixed effects covariance_type must be nonrobust, HC3, or cluster_entity.")
    independent_variables = list(dict.fromkeys(independent_variables))
    validate_model_variables(dataframe, dependent_variable, independent_variables)
    if entity_variable not in dataframe.columns:
        raise KeyError("Panel entity variable is missing from dataframe: " + entity_variable)
    if entity_variable == dependent_variable or entity_variable in independent_variables:
        raise ValueError("Panel entity variable cannot duplicate the outcome or predictors.")
    if time_variable is not None:
        if time_variable not in dataframe.columns:
            raise KeyError("Panel time variable is missing from dataframe: " + time_variable)
        if time_variable == dependent_variable or time_variable in independent_variables:
            raise ValueError("Panel time variable cannot duplicate the outcome or predictors.")

    requested = [dependent_variable, *independent_variables, entity_variable]
    if time_variable is not None:
        requested.append(time_variable)
    work = dataframe[requested].copy()
    work[dependent_variable] = pd.to_numeric(work[dependent_variable], errors="coerce")
    for variable in independent_variables:
        work[variable] = pd.to_numeric(work[variable], errors="coerce")
    work = work.dropna()
    if work.empty:
        raise ValueError("Panel fixed effects has no complete observations to estimate.")
    if work[dependent_variable].nunique() <= 1:
        raise ValueError("Panel dependent variable has no variation.")

    entity_counts = work.groupby(entity_variable).size()
    if len(entity_counts) <= 1:
        raise ValueError("Panel fixed effects requires at least two entities.")
    constant_predictors = [variable for variable in independent_variables if work[variable].nunique() <= 1]
    if constant_predictors:
        raise ValueError("Constant predictors are not supported: " + ", ".join(constant_predictors))

    fe_matrix = _fixed_effect_matrix(work, entity_variable, time_variable)
    y = work[dependent_variable].to_numpy(dtype=float)
    x = work[independent_variables].to_numpy(dtype=float)
    y_within = _residualize_against_fixed_effects(y, fe_matrix)
    x_within = np.column_stack(
        [_residualize_against_fixed_effects(x[:, index], fe_matrix) for index in range(x.shape[1])]
    )
    kept = np.asarray([not np.isclose(np.var(x_within[:, index]), 0.0) for index in range(x_within.shape[1])])
    if not kept.all():
        dropped = [name for name, keep in zip(independent_variables, kept, strict=True) if not keep]
        raise ValueError("Predictors with no within-panel variation cannot be estimated: " + ", ".join(dropped))

    model = sm.OLS(y_within, pd.DataFrame(x_within, columns=independent_variables, index=work.index))
    if covariance_type == "cluster_entity":
        fitted = model.fit(cov_type="cluster", cov_kwds={"groups": work[entity_variable].to_numpy()})
        standard_error_type = "cluster_entity"
    elif covariance_type == "HC3":
        fitted = model.fit(cov_type="HC3")
        standard_error_type = "HC3"
    else:
        fitted = model.fit()
        standard_error_type = "nonrobust"

    confidence_intervals = fitted.conf_int()
    coefficients: list[ModelCoefficient] = []
    for term in fitted.params.index:
        coefficients.append(
            ModelCoefficient(
                term=str(term),
                estimate=float(fitted.params[term]),
                standard_error=float(fitted.bse[term]),
                statistic=float(fitted.tvalues[term]),
                p_value=float(fitted.pvalues[term]),
                confidence_interval_lower=float(confidence_intervals.loc[term, 0]),
                confidence_interval_upper=float(confidence_intervals.loc[term, 1]),
            )
        )

    time_count = int(work[time_variable].nunique()) if time_variable is not None else None
    singleton_count = int((entity_counts == 1).sum())
    warnings: list[str] = []
    if singleton_count:
        warnings.append(f"{singleton_count} entities have only one observation.")
    if len(work) <= len(independent_variables) + len(entity_counts):
        warnings.append("The panel sample size is small relative to absorbed effects and predictors.")

    residuals = np.asarray(fitted.resid, dtype=float)
    fitted_values = np.asarray(fitted.fittedvalues, dtype=float)
    return RegressionResult(
        model_id=model_id,
        model_type="panel_fixed_effects",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(len(work)),
        coefficients=coefficients,
        fit_statistics={
            "within_r_squared": float(fitted.rsquared),
            "adjusted_within_r_squared": float(fitted.rsquared_adj),
            "entity_count": int(len(entity_counts)),
            "time_period_count": time_count,
            "singleton_entity_count": singleton_count,
            "average_observations_per_entity": float(entity_counts.mean()),
            "residual_degrees_of_freedom": float(fitted.df_resid),
        },
        converged=True,
        standard_error_type=standard_error_type,
        warnings=warnings,
        metadata={
            "entity_variable": entity_variable,
            "time_variable": time_variable,
            "absorbed_effects": [entity_variable, *([time_variable] if time_variable else [])],
            "row_labels": [str(index) for index in work.index],
            "entity_labels": work[entity_variable].astype(str).tolist(),
            "time_labels": work[time_variable].astype(str).tolist() if time_variable is not None else None,
            "within_outcome": y_within.tolist(),
            "within_predictors": x_within.tolist(),
            "within_predictor_names": independent_variables,
            "within_fitted_values": fitted_values.tolist(),
            "within_residuals": residuals.tolist(),
            "dropped_case_count": len(dataframe) - len(work),
        },
        raw_result=fitted,
    )


def fit_panel_random_effects(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    entity_variable: str,
    time_variable: str | None = None,
    model_id: str = "panel_random_effects_1",
    reml: bool = False,
    maximum_iterations: int = 200,
) -> RegressionResult:
    """Fit a random-intercept panel regression by entity."""
    independent_variables = list(dict.fromkeys(independent_variables))
    validate_model_variables(dataframe, dependent_variable, independent_variables)
    if entity_variable not in dataframe.columns:
        raise KeyError("Panel entity variable is missing from dataframe: " + entity_variable)
    if entity_variable == dependent_variable or entity_variable in independent_variables:
        raise ValueError("Panel entity variable cannot duplicate the outcome or predictors.")
    if time_variable is not None:
        if time_variable not in dataframe.columns:
            raise KeyError("Panel time variable is missing from dataframe: " + time_variable)
        if time_variable == dependent_variable or time_variable in independent_variables:
            raise ValueError("Panel time variable cannot duplicate the outcome or predictors.")

    requested = [dependent_variable, *independent_variables, entity_variable]
    if time_variable is not None:
        requested.append(time_variable)
    work = dataframe[requested].copy()
    work[dependent_variable] = pd.to_numeric(work[dependent_variable], errors="coerce")
    for variable in independent_variables:
        work[variable] = pd.to_numeric(work[variable], errors="coerce")
    work = work.dropna()
    if work.empty:
        raise ValueError("Panel random effects has no complete observations to estimate.")
    if work[dependent_variable].nunique() <= 1:
        raise ValueError("Panel dependent variable has no variation.")

    entity_counts = work.groupby(entity_variable).size()
    if len(entity_counts) <= 1:
        raise ValueError("Panel random effects requires at least two entities.")
    constant_predictors = [variable for variable in independent_variables if work[variable].nunique() <= 1]
    if constant_predictors:
        raise ValueError("Constant predictors are not supported: " + ", ".join(constant_predictors))

    outcome = work[dependent_variable].to_numpy(dtype=float)
    predictors = sm.add_constant(work[independent_variables], has_constant="add")
    model = sm.MixedLM(outcome, predictors, groups=work[entity_variable].astype(str).to_numpy())
    fitted = model.fit(reml=reml, method="lbfgs", maxiter=maximum_iterations, disp=False)
    confidence_intervals = fitted.conf_int()
    coefficients: list[ModelCoefficient] = []
    bse_fe = np.asarray(fitted.bse_fe, dtype=float)
    for index, term in enumerate(fitted.fe_params.index):
        coefficients.append(
            ModelCoefficient(
                term=str(term),
                estimate=float(fitted.fe_params[term]),
                standard_error=float(bse_fe[index]),
                statistic=float(fitted.tvalues[term]),
                p_value=float(fitted.pvalues[term]),
                confidence_interval_lower=float(confidence_intervals.loc[term, 0]),
                confidence_interval_upper=float(confidence_intervals.loc[term, 1]),
            )
        )

    fixed_fitted = predictors.to_numpy(dtype=float) @ np.asarray(fitted.fe_params, dtype=float)
    fitted_values = np.asarray(fitted.fittedvalues, dtype=float)
    residuals = np.asarray(fitted.resid, dtype=float)
    random_intercept_variance = float(fitted.cov_re.iloc[0, 0]) if fitted.cov_re.size else 0.0
    residual_variance = float(fitted.scale)
    fixed_variance = float(np.var(fixed_fitted, ddof=1)) if len(fixed_fitted) > 1 else 0.0
    total_variance = fixed_variance + random_intercept_variance + residual_variance
    marginal_r_squared = fixed_variance / total_variance if total_variance > 0 else np.nan
    conditional_r_squared = (
        (fixed_variance + random_intercept_variance) / total_variance if total_variance > 0 else np.nan
    )
    time_count = int(work[time_variable].nunique()) if time_variable is not None else None
    singleton_count = int((entity_counts == 1).sum())
    warnings: list[str] = []
    if singleton_count:
        warnings.append(f"{singleton_count} entities have only one observation.")
    if random_intercept_variance <= 1e-10:
        warnings.append("Estimated random-intercept variance is near zero.")
    if not bool(getattr(fitted, "converged", True)):
        warnings.append("Panel random effects model did not converge.")

    return RegressionResult(
        model_id=model_id,
        model_type="panel_random_effects",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(len(work)),
        coefficients=coefficients,
        fit_statistics={
            "entity_count": int(len(entity_counts)),
            "time_period_count": time_count,
            "singleton_entity_count": singleton_count,
            "average_observations_per_entity": float(entity_counts.mean()),
            "random_intercept_variance": random_intercept_variance,
            "residual_variance": residual_variance,
            "marginal_r_squared": float(marginal_r_squared),
            "conditional_r_squared": float(conditional_r_squared),
            "log_likelihood": float(fitted.llf),
            "aic": float(fitted.aic) if not reml else None,
            "bic": float(fitted.bic) if not reml else None,
        },
        converged=bool(getattr(fitted, "converged", True)),
        standard_error_type="mixedlm_model_based",
        warnings=warnings,
        metadata={
            "entity_variable": entity_variable,
            "time_variable": time_variable,
            "reml": reml,
            "maximum_iterations": maximum_iterations,
            "row_labels": [str(index) for index in work.index],
            "entity_labels": work[entity_variable].astype(str).tolist(),
            "time_labels": work[time_variable].astype(str).tolist() if time_variable is not None else None,
            "within_outcome": outcome.tolist(),
            "within_predictors": work[independent_variables].to_numpy(dtype=float).tolist(),
            "within_predictor_names": independent_variables,
            "within_fitted_values": fitted_values.tolist(),
            "within_residuals": residuals.tolist(),
            "fixed_fitted_values": fixed_fitted.tolist(),
            "dropped_case_count": len(dataframe) - len(work),
        },
        raw_result=fitted,
    )


def fit_panel_correlated_random_effects(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    entity_variable: str,
    time_variable: str | None = None,
    model_id: str = "panel_correlated_random_effects_1",
    reml: bool = False,
    maximum_iterations: int = 200,
) -> RegressionResult:
    """Fit a Mundlak correlated random-effects panel regression."""
    independent_variables = list(dict.fromkeys(independent_variables))
    validate_model_variables(dataframe, dependent_variable, independent_variables)
    if entity_variable not in dataframe.columns:
        raise KeyError("Panel entity variable is missing from dataframe: " + entity_variable)
    if entity_variable == dependent_variable or entity_variable in independent_variables:
        raise ValueError("Panel entity variable cannot duplicate the outcome or predictors.")
    if time_variable is not None:
        if time_variable not in dataframe.columns:
            raise KeyError("Panel time variable is missing from dataframe: " + time_variable)
        if time_variable == dependent_variable or time_variable in independent_variables:
            raise ValueError("Panel time variable cannot duplicate the outcome or predictors.")

    requested = [dependent_variable, *independent_variables, entity_variable]
    if time_variable is not None:
        requested.append(time_variable)
    work = dataframe[requested].copy()
    work[dependent_variable] = pd.to_numeric(work[dependent_variable], errors="coerce")
    for variable in independent_variables:
        work[variable] = pd.to_numeric(work[variable], errors="coerce")
    work = work.dropna()
    if work.empty:
        raise ValueError("Panel correlated random effects has no complete observations to estimate.")
    if work[dependent_variable].nunique() <= 1:
        raise ValueError("Panel dependent variable has no variation.")

    entity_counts = work.groupby(entity_variable).size()
    if len(entity_counts) <= 1:
        raise ValueError("Panel correlated random effects requires at least two entities.")
    constant_predictors = [variable for variable in independent_variables if work[variable].nunique() <= 1]
    if constant_predictors:
        raise ValueError("Constant predictors are not supported: " + ", ".join(constant_predictors))

    entity_mean_terms: list[str] = []
    dropped_mean_terms: list[str] = []
    for variable in independent_variables:
        base_name = f"mean_{variable}"
        mean_name = base_name
        suffix = 2
        while mean_name in work.columns or mean_name in entity_mean_terms:
            mean_name = f"{base_name}_{suffix}"
            suffix += 1
        values = work.groupby(entity_variable, sort=False)[variable].transform("mean")
        if np.isclose(float(values.var(ddof=1)), 0.0):
            dropped_mean_terms.append(mean_name)
            continue
        work[mean_name] = values
        entity_mean_terms.append(mean_name)

    model_predictors = [*independent_variables, *entity_mean_terms]
    outcome = work[dependent_variable].to_numpy(dtype=float)
    predictors = sm.add_constant(work[model_predictors], has_constant="add")
    model = sm.MixedLM(outcome, predictors, groups=work[entity_variable].astype(str).to_numpy())
    fitted = model.fit(reml=reml, method="lbfgs", maxiter=maximum_iterations, disp=False)
    confidence_intervals = fitted.conf_int()
    coefficients: list[ModelCoefficient] = []
    bse_fe = np.asarray(fitted.bse_fe, dtype=float)
    for index, term in enumerate(fitted.fe_params.index):
        coefficients.append(
            ModelCoefficient(
                term=str(term),
                estimate=float(fitted.fe_params[term]),
                standard_error=float(bse_fe[index]),
                statistic=float(fitted.tvalues[term]),
                p_value=float(fitted.pvalues[term]),
                confidence_interval_lower=float(confidence_intervals.loc[term, 0]),
                confidence_interval_upper=float(confidence_intervals.loc[term, 1]),
            )
        )

    fixed_fitted = predictors.to_numpy(dtype=float) @ np.asarray(fitted.fe_params, dtype=float)
    fitted_values = np.asarray(fitted.fittedvalues, dtype=float)
    residuals = np.asarray(fitted.resid, dtype=float)
    random_intercept_variance = float(fitted.cov_re.iloc[0, 0]) if fitted.cov_re.size else 0.0
    residual_variance = float(fitted.scale)
    fixed_variance = float(np.var(fixed_fitted, ddof=1)) if len(fixed_fitted) > 1 else 0.0
    total_variance = fixed_variance + random_intercept_variance + residual_variance
    marginal_r_squared = fixed_variance / total_variance if total_variance > 0 else np.nan
    conditional_r_squared = (
        (fixed_variance + random_intercept_variance) / total_variance if total_variance > 0 else np.nan
    )
    time_count = int(work[time_variable].nunique()) if time_variable is not None else None
    singleton_count = int((entity_counts == 1).sum())
    warnings: list[str] = []
    if singleton_count:
        warnings.append(f"{singleton_count} entities have only one observation.")
    if dropped_mean_terms:
        warnings.append("Entity-mean terms with no between-entity variation were dropped.")
    if random_intercept_variance <= 1e-10:
        warnings.append("Estimated random-intercept variance is near zero.")
    if not bool(getattr(fitted, "converged", True)):
        warnings.append("Panel correlated random effects model did not converge.")

    return RegressionResult(
        model_id=model_id,
        model_type="panel_correlated_random_effects",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(len(work)),
        coefficients=coefficients,
        fit_statistics={
            "entity_count": int(len(entity_counts)),
            "time_period_count": time_count,
            "singleton_entity_count": singleton_count,
            "average_observations_per_entity": float(entity_counts.mean()),
            "random_intercept_variance": random_intercept_variance,
            "residual_variance": residual_variance,
            "marginal_r_squared": float(marginal_r_squared),
            "conditional_r_squared": float(conditional_r_squared),
            "entity_mean_term_count": len(entity_mean_terms),
            "log_likelihood": float(fitted.llf),
            "aic": float(fitted.aic) if not reml else None,
            "bic": float(fitted.bic) if not reml else None,
        },
        converged=bool(getattr(fitted, "converged", True)),
        standard_error_type="mixedlm_model_based",
        warnings=warnings,
        metadata={
            "entity_variable": entity_variable,
            "time_variable": time_variable,
            "reml": reml,
            "maximum_iterations": maximum_iterations,
            "entity_mean_terms": entity_mean_terms,
            "dropped_entity_mean_terms": dropped_mean_terms,
            "row_labels": [str(index) for index in work.index],
            "entity_labels": work[entity_variable].astype(str).tolist(),
            "time_labels": work[time_variable].astype(str).tolist() if time_variable is not None else None,
            "within_outcome": outcome.tolist(),
            "within_predictors": work[model_predictors].to_numpy(dtype=float).tolist(),
            "within_predictor_names": model_predictors,
            "within_fitted_values": fitted_values.tolist(),
            "within_residuals": residuals.tolist(),
            "fixed_fitted_values": fixed_fitted.tolist(),
            "dropped_case_count": len(dataframe) - len(work),
            "mundlak_correction": True,
        },
        raw_result=fitted,
    )


def fit_panel_between_effects(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    entity_variable: str,
    time_variable: str | None = None,
    model_id: str = "panel_between_effects_1",
    covariance_type: str = "HC3",
) -> RegressionResult:
    """Fit a between-effects panel regression on entity-level means."""
    if covariance_type not in {"nonrobust", "HC3"}:
        raise ValueError("Panel between effects covariance_type must be nonrobust or HC3.")
    independent_variables = list(dict.fromkeys(independent_variables))
    validate_model_variables(dataframe, dependent_variable, independent_variables)
    if entity_variable not in dataframe.columns:
        raise KeyError("Panel entity variable is missing from dataframe: " + entity_variable)
    if entity_variable == dependent_variable or entity_variable in independent_variables:
        raise ValueError("Panel entity variable cannot duplicate the outcome or predictors.")
    if time_variable is not None:
        if time_variable not in dataframe.columns:
            raise KeyError("Panel time variable is missing from dataframe: " + time_variable)
        if time_variable == dependent_variable or time_variable in independent_variables:
            raise ValueError("Panel time variable cannot duplicate the outcome or predictors.")

    requested = [dependent_variable, *independent_variables, entity_variable]
    if time_variable is not None:
        requested.append(time_variable)
    work = dataframe[requested].copy()
    work[dependent_variable] = pd.to_numeric(work[dependent_variable], errors="coerce")
    for variable in independent_variables:
        work[variable] = pd.to_numeric(work[variable], errors="coerce")
    work = work.dropna()
    if work.empty:
        raise ValueError("Panel between effects has no complete observations to estimate.")

    entity_counts = work.groupby(entity_variable).size()
    if len(entity_counts) <= 1:
        raise ValueError("Panel between effects requires at least two entities.")
    entity_means = work.groupby(entity_variable, sort=False)[[dependent_variable, *independent_variables]].mean()
    if entity_means[dependent_variable].nunique() <= 1:
        raise ValueError("Between-entity dependent variable means have no variation.")
    constant_predictors = [variable for variable in independent_variables if entity_means[variable].nunique() <= 1]
    if constant_predictors:
        raise ValueError("Predictors with no between-entity variation cannot be estimated: " + ", ".join(constant_predictors))

    outcome = entity_means[dependent_variable].to_numpy(dtype=float)
    predictors = sm.add_constant(entity_means[independent_variables], has_constant="add")
    model = sm.OLS(outcome, predictors)
    fitted = model.fit(cov_type="HC3") if covariance_type == "HC3" else model.fit()
    confidence_intervals = fitted.conf_int()

    coefficients: list[ModelCoefficient] = []
    for term in fitted.params.index:
        coefficients.append(
            ModelCoefficient(
                term=str(term),
                estimate=float(fitted.params[term]),
                standard_error=float(fitted.bse[term]),
                statistic=float(fitted.tvalues[term]),
                p_value=float(fitted.pvalues[term]),
                confidence_interval_lower=float(confidence_intervals.loc[term, 0]),
                confidence_interval_upper=float(confidence_intervals.loc[term, 1]),
            )
        )

    fitted_values = np.asarray(fitted.fittedvalues, dtype=float)
    residuals = np.asarray(fitted.resid, dtype=float)
    time_count = int(work[time_variable].nunique()) if time_variable is not None else None
    singleton_count = int((entity_counts == 1).sum())
    warnings: list[str] = []
    if singleton_count:
        warnings.append(f"{singleton_count} entities have only one observation.")
    if len(entity_means) <= len(independent_variables) + 1:
        warnings.append("The number of entities is small relative to the number of predictors.")

    entity_labels = [str(value) for value in entity_means.index.tolist()]
    return RegressionResult(
        model_id=model_id,
        model_type="panel_between_effects",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(len(entity_means)),
        coefficients=coefficients,
        fit_statistics={
            "between_r_squared": float(fitted.rsquared),
            "adjusted_between_r_squared": float(fitted.rsquared_adj),
            "entity_count": int(len(entity_counts)),
            "time_period_count": time_count,
            "singleton_entity_count": singleton_count,
            "average_observations_per_entity": float(entity_counts.mean()),
            "overall_observation_count": int(len(work)),
            "residual_degrees_of_freedom": float(fitted.df_resid),
            "aic": float(fitted.aic),
            "bic": float(fitted.bic),
        },
        converged=True,
        standard_error_type=covariance_type,
        warnings=warnings,
        metadata={
            "entity_variable": entity_variable,
            "time_variable": time_variable,
            "row_labels": entity_labels,
            "entity_labels": entity_labels,
            "time_labels": None,
            "within_outcome": outcome.tolist(),
            "within_predictors": entity_means[independent_variables].to_numpy(dtype=float).tolist(),
            "within_predictor_names": independent_variables,
            "within_fitted_values": fitted_values.tolist(),
            "within_residuals": residuals.tolist(),
            "between_entity_means": entity_means.reset_index().to_dict(orient="list"),
            "dropped_case_count": len(dataframe) - len(work),
        },
        raw_result=fitted,
    )


def fit_panel_first_difference(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    entity_variable: str,
    time_variable: str,
    model_id: str = "panel_first_difference_1",
    covariance_type: str = "HC3",
    add_intercept: bool = False,
) -> RegressionResult:
    """Fit a first-difference panel regression using within-entity changes."""
    if covariance_type not in {"nonrobust", "HC3", "cluster_entity"}:
        raise ValueError("Panel first difference covariance_type must be nonrobust, HC3, or cluster_entity.")
    independent_variables = list(dict.fromkeys(independent_variables))
    validate_model_variables(dataframe, dependent_variable, independent_variables)
    if entity_variable not in dataframe.columns:
        raise KeyError("Panel entity variable is missing from dataframe: " + entity_variable)
    if time_variable not in dataframe.columns:
        raise KeyError("Panel first difference requires a time variable in the dataframe: " + time_variable)
    if entity_variable == dependent_variable or entity_variable in independent_variables:
        raise ValueError("Panel entity variable cannot duplicate the outcome or predictors.")
    if time_variable == dependent_variable or time_variable in independent_variables:
        raise ValueError("Panel time variable cannot duplicate the outcome or predictors.")

    requested = [dependent_variable, *independent_variables, entity_variable, time_variable]
    work = dataframe[requested].copy()
    work[dependent_variable] = pd.to_numeric(work[dependent_variable], errors="coerce")
    for variable in independent_variables:
        work[variable] = pd.to_numeric(work[variable], errors="coerce")
    work = work.dropna().sort_values([entity_variable, time_variable])
    if work.empty:
        raise ValueError("Panel first difference has no complete observations to estimate.")

    entity_counts = work.groupby(entity_variable).size()
    if len(entity_counts) <= 1:
        raise ValueError("Panel first difference requires at least two entities.")
    differenced = work.copy()
    for variable in [dependent_variable, *independent_variables]:
        differenced[variable] = work.groupby(entity_variable, sort=False)[variable].diff()
    differenced = differenced.dropna(subset=[dependent_variable, *independent_variables])
    if differenced.empty:
        raise ValueError("Panel first difference requires at least two complete observations per entity.")
    used_entities = differenced[entity_variable].nunique()
    if used_entities <= 1:
        raise ValueError("Panel first difference requires differenced observations from at least two entities.")
    if differenced[dependent_variable].nunique() <= 1:
        raise ValueError("First-differenced dependent variable has no variation.")
    constant_predictors = [
        variable for variable in independent_variables if np.isclose(differenced[variable].var(ddof=1), 0.0)
    ]
    if constant_predictors:
        raise ValueError(
            "Predictors with no first-difference variation cannot be estimated: " + ", ".join(constant_predictors)
        )

    outcome = differenced[dependent_variable].to_numpy(dtype=float)
    predictors = differenced[independent_variables].copy()
    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")
    model = sm.OLS(outcome, predictors)
    if covariance_type == "cluster_entity":
        fitted = model.fit(cov_type="cluster", cov_kwds={"groups": differenced[entity_variable].to_numpy()})
        standard_error_type = "cluster_entity"
    elif covariance_type == "HC3":
        fitted = model.fit(cov_type="HC3")
        standard_error_type = "HC3"
    else:
        fitted = model.fit()
        standard_error_type = "nonrobust"
    confidence_intervals = fitted.conf_int()

    coefficients: list[ModelCoefficient] = []
    for term in fitted.params.index:
        coefficients.append(
            ModelCoefficient(
                term=str(term),
                estimate=float(fitted.params[term]),
                standard_error=float(fitted.bse[term]),
                statistic=float(fitted.tvalues[term]),
                p_value=float(fitted.pvalues[term]),
                confidence_interval_lower=float(confidence_intervals.loc[term, 0]),
                confidence_interval_upper=float(confidence_intervals.loc[term, 1]),
            )
        )

    fitted_values = np.asarray(fitted.fittedvalues, dtype=float)
    residuals = np.asarray(fitted.resid, dtype=float)
    singleton_count = int((entity_counts == 1).sum())
    time_count = int(work[time_variable].nunique())
    differenced_entity_counts = differenced.groupby(entity_variable).size()
    warnings: list[str] = []
    dropped_entity_count = int(len(entity_counts) - used_entities)
    if dropped_entity_count:
        warnings.append(f"{dropped_entity_count} entities were dropped because they had no differenced observations.")
    if len(differenced) <= len(predictors.columns) + 1:
        warnings.append("The first-difference sample size is small relative to the number of predictors.")

    return RegressionResult(
        model_id=model_id,
        model_type="panel_first_difference",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(len(differenced)),
        coefficients=coefficients,
        fit_statistics={
            "first_difference_r_squared": float(fitted.rsquared),
            "adjusted_first_difference_r_squared": float(fitted.rsquared_adj),
            "entity_count": int(len(entity_counts)),
            "differenced_entity_count": int(used_entities),
            "time_period_count": time_count,
            "singleton_entity_count": singleton_count,
            "average_differenced_observations_per_entity": float(differenced_entity_counts.mean()),
            "overall_observation_count": int(len(work)),
            "residual_degrees_of_freedom": float(fitted.df_resid),
            "aic": float(fitted.aic),
            "bic": float(fitted.bic),
        },
        converged=True,
        standard_error_type=standard_error_type,
        warnings=warnings,
        metadata={
            "entity_variable": entity_variable,
            "time_variable": time_variable,
            "add_intercept": add_intercept,
            "row_labels": [str(index) for index in differenced.index],
            "entity_labels": differenced[entity_variable].astype(str).tolist(),
            "time_labels": differenced[time_variable].astype(str).tolist(),
            "within_outcome": outcome.tolist(),
            "within_predictors": differenced[independent_variables].to_numpy(dtype=float).tolist(),
            "within_predictor_names": independent_variables,
            "within_fitted_values": fitted_values.tolist(),
            "within_residuals": residuals.tolist(),
            "first_difference": True,
            "dropped_case_count": len(dataframe) - len(work),
        },
        raw_result=fitted,
    )


def fit_panel_pooled_ols(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    entity_variable: str,
    time_variable: str | None = None,
    model_id: str = "panel_pooled_ols_1",
    covariance_type: str = "cluster_entity",
    add_intercept: bool = True,
) -> RegressionResult:
    """Fit pooled OLS for panel data with optional entity-clustered covariance."""
    if covariance_type not in {"nonrobust", "HC3", "cluster_entity"}:
        raise ValueError("Panel pooled OLS covariance_type must be nonrobust, HC3, or cluster_entity.")
    independent_variables = list(dict.fromkeys(independent_variables))
    validate_model_variables(dataframe, dependent_variable, independent_variables)
    if entity_variable not in dataframe.columns:
        raise KeyError("Panel entity variable is missing from dataframe: " + entity_variable)
    if entity_variable == dependent_variable or entity_variable in independent_variables:
        raise ValueError("Panel entity variable cannot duplicate the outcome or predictors.")
    if time_variable is not None:
        if time_variable not in dataframe.columns:
            raise KeyError("Panel time variable is missing from dataframe: " + time_variable)
        if time_variable == dependent_variable or time_variable in independent_variables:
            raise ValueError("Panel time variable cannot duplicate the outcome or predictors.")

    requested = [dependent_variable, *independent_variables, entity_variable]
    if time_variable is not None:
        requested.append(time_variable)
    work = dataframe[requested].copy()
    work[dependent_variable] = pd.to_numeric(work[dependent_variable], errors="coerce")
    for variable in independent_variables:
        work[variable] = pd.to_numeric(work[variable], errors="coerce")
    work = work.dropna()
    if work.empty:
        raise ValueError("Panel pooled OLS has no complete observations to estimate.")
    if work[dependent_variable].nunique() <= 1:
        raise ValueError("Panel dependent variable has no variation.")

    entity_counts = work.groupby(entity_variable).size()
    if len(entity_counts) <= 1:
        raise ValueError("Panel pooled OLS requires at least two entities.")
    constant_predictors = [variable for variable in independent_variables if work[variable].nunique() <= 1]
    if constant_predictors:
        raise ValueError("Constant predictors are not supported: " + ", ".join(constant_predictors))

    outcome = work[dependent_variable].to_numpy(dtype=float)
    predictors = work[independent_variables].copy()
    if add_intercept:
        predictors = sm.add_constant(predictors, has_constant="add")
    model = sm.OLS(outcome, predictors)
    if covariance_type == "cluster_entity":
        fitted = model.fit(cov_type="cluster", cov_kwds={"groups": work[entity_variable].to_numpy()})
        standard_error_type = "cluster_entity"
    elif covariance_type == "HC3":
        fitted = model.fit(cov_type="HC3")
        standard_error_type = "HC3"
    else:
        fitted = model.fit()
        standard_error_type = "nonrobust"
    confidence_intervals = fitted.conf_int()

    coefficients: list[ModelCoefficient] = []
    for term in fitted.params.index:
        coefficients.append(
            ModelCoefficient(
                term=str(term),
                estimate=float(fitted.params[term]),
                standard_error=float(fitted.bse[term]),
                statistic=float(fitted.tvalues[term]),
                p_value=float(fitted.pvalues[term]),
                confidence_interval_lower=float(confidence_intervals.loc[term, 0]),
                confidence_interval_upper=float(confidence_intervals.loc[term, 1]),
            )
        )

    fitted_values = np.asarray(fitted.fittedvalues, dtype=float)
    residuals = np.asarray(fitted.resid, dtype=float)
    singleton_count = int((entity_counts == 1).sum())
    time_count = int(work[time_variable].nunique()) if time_variable is not None else None
    warnings: list[str] = []
    if singleton_count:
        warnings.append(f"{singleton_count} entities have only one observation.")
    if covariance_type == "cluster_entity" and len(entity_counts) < 10:
        warnings.append("Cluster-robust standard errors can be unstable with fewer than 10 entities.")

    return RegressionResult(
        model_id=model_id,
        model_type="panel_pooled_ols",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(len(work)),
        coefficients=coefficients,
        fit_statistics={
            "pooled_r_squared": float(fitted.rsquared),
            "adjusted_pooled_r_squared": float(fitted.rsquared_adj),
            "entity_count": int(len(entity_counts)),
            "time_period_count": time_count,
            "singleton_entity_count": singleton_count,
            "average_observations_per_entity": float(entity_counts.mean()),
            "residual_degrees_of_freedom": float(fitted.df_resid),
            "aic": float(fitted.aic),
            "bic": float(fitted.bic),
        },
        converged=True,
        standard_error_type=standard_error_type,
        warnings=warnings,
        metadata={
            "entity_variable": entity_variable,
            "time_variable": time_variable,
            "add_intercept": add_intercept,
            "row_labels": [str(index) for index in work.index],
            "entity_labels": work[entity_variable].astype(str).tolist(),
            "time_labels": work[time_variable].astype(str).tolist() if time_variable is not None else None,
            "within_outcome": outcome.tolist(),
            "within_predictors": work[independent_variables].to_numpy(dtype=float).tolist(),
            "within_predictor_names": independent_variables,
            "within_fitted_values": fitted_values.tolist(),
            "within_residuals": residuals.tolist(),
            "pooled": True,
            "dropped_case_count": len(dataframe) - len(work),
        },
        raw_result=fitted,
    )
