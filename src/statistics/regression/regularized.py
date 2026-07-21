"""Regularized linear regression with ridge, lasso, and elastic-net penalties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize

from src.statistics.regression.base import ModelCoefficient, RegressionResult
from src.statistics.regression.design_matrix import prepare_regression_design_matrix


@dataclass(slots=True)
class RegularizedRawResult:
    params: pd.Series
    fittedvalues: pd.Series
    resid: pd.Series
    model: Any
    alpha: float
    l1_ratio: float
    standardized_coefficients: pd.Series
    predictor_means: pd.Series
    predictor_scales: pd.Series
    converged: bool
    message: str


def _l1_ratio_from_penalty(penalty: str, l1_ratio: float) -> float:
    key = penalty.strip().lower().replace("-", "_")
    if key == "ridge":
        return 0.0
    if key == "lasso":
        return 1.0
    if key in {"elastic_net", "elasticnet"}:
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError("Elastic-net l1_ratio must be between 0 and 1.")
        return float(l1_ratio)
    raise ValueError("Regularized regression penalty must be ridge, lasso, or elastic_net.")


def _standardize_predictors(predictors: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    means = predictors.mean(axis=0)
    scales = predictors.std(axis=0, ddof=0).replace(0.0, 1.0)
    standardized = (predictors - means) / scales
    return standardized, means, scales


def fit_regularized_regression(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    fixed_effects: list[str] | None = None,
    model_id: str = "regularized_regression_1",
    penalty: str = "elastic_net",
    alpha: float = 0.1,
    l1_ratio: float = 0.5,
    add_intercept: bool = True,
    standardize: bool = True,
    maximum_iterations: int = 1000,
) -> RegressionResult:
    """Fit a penalized least-squares model."""
    if alpha < 0:
        raise ValueError("Regularized regression alpha must be non-negative.")
    resolved_l1_ratio = _l1_ratio_from_penalty(penalty, l1_ratio)
    independent_variables = list(dict.fromkeys(independent_variables))
    fixed_effects = list(dict.fromkeys(fixed_effects or []))
    design = prepare_regression_design_matrix(
        dataframe,
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        fixed_effects=fixed_effects,
        model_label="Regularized regression",
    )
    outcome = design.outcome.astype(float)
    predictors = design.predictors.astype(float)
    if standardize:
        model_predictors, means, scales = _standardize_predictors(predictors)
    else:
        model_predictors = predictors.copy()
        means = pd.Series(0.0, index=predictors.columns)
        scales = pd.Series(1.0, index=predictors.columns)

    y = outcome.to_numpy(dtype=float)
    x = model_predictors.to_numpy(dtype=float)
    y_mean = float(np.mean(y)) if add_intercept else 0.0
    centered_y = y - y_mean if add_intercept else y.copy()
    parameter_count = x.shape[1]

    def objective(beta: np.ndarray) -> float:
        residual = centered_y - x @ beta
        loss = 0.5 * float(np.mean(residual**2))
        ridge = 0.5 * (1.0 - resolved_l1_ratio) * float(np.sum(beta**2))
        lasso = resolved_l1_ratio * float(np.sum(np.abs(beta)))
        return loss + float(alpha) * (ridge + lasso)

    start = np.linalg.lstsq(x, centered_y, rcond=None)[0]
    fitted = optimize.minimize(
        objective,
        start,
        method="L-BFGS-B",
        options={"maxiter": maximum_iterations},
    )
    standardized_beta = np.asarray(fitted.x, dtype=float)
    original_beta = standardized_beta / scales.to_numpy(dtype=float)
    intercept = y_mean - float(np.dot(means.to_numpy(dtype=float), original_beta)) if add_intercept else 0.0
    fitted_values = intercept + predictors.to_numpy(dtype=float) @ original_beta
    residuals = y - fitted_values
    names = [str(column) for column in predictors.columns]
    params = pd.Series(original_beta, index=names, dtype=float)
    if add_intercept:
        params = pd.concat([pd.Series({"const": intercept}, dtype=float), params])
    standardized_series = pd.Series(standardized_beta, index=names, dtype=float)
    row_labels = outcome.index.tolist()
    raw_result = RegularizedRawResult(
        params=params,
        fittedvalues=pd.Series(fitted_values, index=outcome.index),
        resid=pd.Series(residuals, index=outcome.index),
        model=type(
            "RegularizedModelData",
            (),
            {
                "endog": y,
                "exog": sm.add_constant(predictors, has_constant="add").to_numpy(dtype=float)
                if add_intercept
                else predictors.to_numpy(dtype=float),
                "exog_names": ["const", *names] if add_intercept else names,
                "data": type("RegularizedModelRows", (), {"row_labels": row_labels})(),
            },
        )(),
        alpha=float(alpha),
        l1_ratio=float(resolved_l1_ratio),
        standardized_coefficients=standardized_series,
        predictor_means=means,
        predictor_scales=scales,
        converged=bool(fitted.success),
        message=str(fitted.message),
    )

    coefficients: list[ModelCoefficient] = []
    for term, estimate in params.items():
        coefficients.append(
            ModelCoefficient(
                term=str(term),
                estimate=float(estimate),
                standard_error=np.nan,
                statistic=np.nan,
                p_value=np.nan,
                confidence_interval_lower=np.nan,
                confidence_interval_upper=np.nan,
            )
        )

    abs_beta = np.abs(original_beta)
    selected_mask = abs_beta > 1e-8
    ss_resid = float(np.sum(residuals**2))
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    pseudo_r_squared = 1.0 - ss_resid / ss_total if ss_total > 0 else np.nan
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    warnings: list[str] = []
    if not fitted.success:
        warnings.append("Regularized regression optimization did not fully converge.")
    if selected_mask.sum() == 0:
        warnings.append("All penalized coefficients were shrunk to zero.")

    return RegressionResult(
        model_id=model_id,
        model_type="regularized_regression",
        dependent_variable=dependent_variable,
        independent_variables=independent_variables,
        sample_size=int(len(outcome)),
        coefficients=coefficients,
        fit_statistics={
            "penalty": penalty,
            "alpha": float(alpha),
            "l1_ratio": float(resolved_l1_ratio),
            "pseudo_r_squared": float(pseudo_r_squared),
            "root_mean_squared_error": rmse,
            "mean_absolute_error": mae,
            "selected_coefficient_count": int(selected_mask.sum()),
            "zero_coefficient_count": int((~selected_mask).sum()),
            "parameter_count": int(parameter_count + (1 if add_intercept else 0)),
            "objective_value": float(fitted.fun),
        },
        converged=bool(fitted.success),
        standard_error_type="penalized_no_inference",
        warnings=warnings,
        metadata={
            "penalty": penalty,
            "alpha": float(alpha),
            "l1_ratio": float(resolved_l1_ratio),
            "add_intercept": add_intercept,
            "standardize": standardize,
            "design_matrix_columns": ["const", *names] if add_intercept else names,
            "standardized_coefficients": standardized_series.to_dict(),
            "selected_terms": [name for name, keep in zip(names, selected_mask, strict=True) if keep],
            "zero_terms": [name for name, keep in zip(names, selected_mask, strict=True) if not keep],
            "row_labels": [str(index) for index in outcome.index],
            "fitted_values": fitted_values.tolist(),
            "residuals": residuals.tolist(),
            **design.metadata,
        },
        raw_result=raw_result,
    )
