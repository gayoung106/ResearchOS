"""종속변수 측정수준과 명시적 모형 설정에 따라 회귀모형을 선택한다."""

from __future__ import annotations

import pandas as pd

from src.statistics.regression.base import RegressionResult
from src.statistics.regression.beta import fit_beta_regression
from src.statistics.regression.binary_cloglog import fit_binary_cloglog
from src.statistics.regression.binary_logit import fit_binary_logit
from src.statistics.regression.binary_probit import fit_binary_probit
from src.statistics.regression.count import fit_count_regression
from src.statistics.regression.cox import fit_cox_proportional_hazards
from src.statistics.regression.exponential_aft import fit_exponential_aft
from src.statistics.regression.fractional_logit import fit_fractional_logit
from src.statistics.regression.gamma import fit_gamma_regression
from src.statistics.regression.gee import fit_gee
from src.statistics.regression.heckman import fit_heckman_selection
from src.statistics.regression.inverse_gaussian import fit_inverse_gaussian_regression
from src.statistics.regression.iv import fit_iv_2sls_regression
from src.statistics.regression.log_binomial import fit_log_binomial
from src.statistics.regression.loglogistic_aft import fit_loglogistic_aft
from src.statistics.regression.lognormal_aft import fit_lognormal_aft
from src.statistics.regression.mixed_binary_logit import (
    fit_mixed_binary_logit_random_intercept,
    fit_mixed_binary_logit_random_slope,
    fit_mixed_binary_logit_three_level,
)
from src.statistics.regression.mixed_count import (
    fit_mixed_poisson_random_intercept,
    fit_mixed_poisson_random_slope,
    fit_mixed_poisson_three_level,
)
from src.statistics.regression.mixed_effects import (
    fit_multiple_random_slopes,
    fit_random_intercept,
    fit_random_slope,
    fit_three_level_mixed_effects,
)
from src.statistics.regression.mixed_negative_binomial import (
    fit_mixed_negative_binomial_random_intercept,
    fit_mixed_negative_binomial_random_slope,
    fit_mixed_negative_binomial_three_level,
)
from src.statistics.regression.multinomial_logit import fit_multinomial_logit
from src.statistics.regression.ols import fit_ols
from src.statistics.regression.ordered_logit import fit_ordered_logit
from src.statistics.regression.ordered_probit import fit_ordered_probit
from src.statistics.regression.panel import fit_panel_fixed_effects
from src.statistics.regression.quantile import fit_quantile_regression
from src.statistics.regression.regularized import fit_regularized_regression
from src.statistics.regression.robust import fit_robust_regression
from src.statistics.regression.tobit import fit_tobit_regression
from src.statistics.regression.weibull_aft import fit_weibull_aft
from src.statistics.regression.weighted_least_squares import fit_weighted_least_squares


def fit_regression_by_level(
    dataframe: pd.DataFrame,
    *,
    dependent_variable: str,
    independent_variables: list[str],
    measurement_level: str,
    fixed_effects: list[str] | None = None,
    model_id: str = "model_1",
    model_type: str | None = None,
    group_variable: str | None = None,
    mixed_effects_options: dict[str, object] | None = None,
) -> RegressionResult:
    """측정수준 또는 명시적 모형 설정에 적합한 회귀모형을 실행한다."""
    if model_type == "log_binomial":
        options = mixed_effects_options or {}
        return fit_log_binomial(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            covariance_type=str(options.get("covariance_type", "HC3")),
            add_intercept=bool(options.get("add_intercept", True)),
            maximum_iterations=int(options.get("max_iterations", options.get("maximum_iterations", 100))),
        )

    if model_type == "weighted_least_squares":
        options = mixed_effects_options or {}
        weight_variable = str(options.get("weight_variable", options.get("weights", ""))).strip()
        if not weight_variable:
            raise ValueError("Weighted least squares requires weight_variable.")
        return fit_weighted_least_squares(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            weight_variable=weight_variable,
            fixed_effects=fixed_effects,
            model_id=model_id,
            covariance_type=str(options.get("covariance_type", "HC3")),
            add_intercept=bool(options.get("add_intercept", True)),
        )

    if model_type == "binary_cloglog":
        options = mixed_effects_options or {}
        return fit_binary_cloglog(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            covariance_type=str(options.get("covariance_type", "HC3")),
            add_intercept=bool(options.get("add_intercept", True)),
            maximum_iterations=int(options.get("max_iterations", options.get("maximum_iterations", 100))),
        )

    if model_type == "ordered_probit":
        options = mixed_effects_options or {}
        return fit_ordered_probit(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            maximum_iterations=int(options.get("max_iterations", options.get("maximum_iterations", 200))),
        )

    if model_type == "binary_probit":
        options = mixed_effects_options or {}
        return fit_binary_probit(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            covariance_type=str(options.get("covariance_type", "HC3")),
            add_intercept=bool(options.get("add_intercept", True)),
            maximum_iterations=int(options.get("max_iterations", options.get("maximum_iterations", 100))),
        )

    if model_type == "heckman_selection":
        options = mixed_effects_options or {}
        selection_variable = str(options.get("selection_variable", options.get("selected_variable", ""))).strip()
        selection_variables = options.get("selection_variables", options.get("selection_predictors", []))
        if not selection_variable:
            raise ValueError("Heckman selection requires selection_variable.")
        return fit_heckman_selection(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            selection_variable=selection_variable,
            selection_variables=[str(value) for value in selection_variables]
            if isinstance(selection_variables, (list, tuple))
            else [str(selection_variables)],
            fixed_effects=fixed_effects,
            model_id=model_id,
            covariance_type=str(options.get("covariance_type", "HC3")),
            add_intercept=bool(options.get("add_intercept", True)),
            maximum_iterations=int(options.get("max_iterations", options.get("maximum_iterations", 100))),
        )

    if model_type == "iv_2sls_regression":
        options = mixed_effects_options or {}
        endogenous = options.get("endogenous_variables", options.get("endogenous", []))
        instruments = options.get("instrument_variables", options.get("instruments", []))
        return fit_iv_2sls_regression(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            endogenous_variables=[str(value) for value in endogenous] if isinstance(endogenous, (list, tuple)) else [str(endogenous)],
            instrument_variables=[str(value) for value in instruments] if isinstance(instruments, (list, tuple)) else [str(instruments)],
            fixed_effects=fixed_effects,
            model_id=model_id,
            add_intercept=bool(options.get("add_intercept", True)),
        )

    if model_type == "inverse_gaussian_regression":
        options = mixed_effects_options or {}
        return fit_inverse_gaussian_regression(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            covariance_type=str(options.get("covariance_type", "HC3")),
            add_intercept=bool(options.get("add_intercept", True)),
            maximum_iterations=int(options.get("max_iterations", options.get("maximum_iterations", 100))),
        )

    if model_type == "gamma_regression":
        options = mixed_effects_options or {}
        return fit_gamma_regression(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            covariance_type=str(options.get("covariance_type", "HC3")),
            add_intercept=bool(options.get("add_intercept", True)),
            maximum_iterations=int(options.get("max_iterations", options.get("maximum_iterations", 100))),
        )

    if model_type == "regularized_regression":
        options = mixed_effects_options or {}
        return fit_regularized_regression(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            penalty=str(options.get("penalty", options.get("regularization", "elastic_net"))),
            alpha=float(options.get("alpha", options.get("lambda", 0.1))),
            l1_ratio=float(options.get("l1_ratio", 0.5)),
            add_intercept=bool(options.get("add_intercept", True)),
            standardize=bool(options.get("standardize", True)),
            maximum_iterations=int(options.get("max_iterations", options.get("maximum_iterations", 1000))),
        )

    if model_type == "robust_regression":
        options = mixed_effects_options or {}
        return fit_robust_regression(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            norm=str(options.get("norm", options.get("robust_norm", "huber"))),
            add_intercept=bool(options.get("add_intercept", True)),
            maximum_iterations=int(options.get("max_iterations", options.get("maximum_iterations", 100))),
        )

    if model_type == "tobit_regression":
        options = mixed_effects_options or {}
        lower_limit = options.get("lower_limit")
        upper_limit = options.get("upper_limit")
        return fit_tobit_regression(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            lower_limit=float(lower_limit) if lower_limit is not None else None,
            upper_limit=float(upper_limit) if upper_limit is not None else None,
            add_intercept=bool(options.get("add_intercept", True)),
            maximum_iterations=int(options.get("max_iterations", options.get("maximum_iterations", 300))),
        )

    if model_type == "panel_fixed_effects":
        options = mixed_effects_options or {}
        entity_variable = str(options.get("entity_variable", options.get("id_variable", ""))).strip()
        time_variable = str(options.get("time_variable", "")).strip() or None
        if not entity_variable:
            raise ValueError("Panel fixed effects requires entity_variable.")
        return fit_panel_fixed_effects(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            entity_variable=entity_variable,
            time_variable=time_variable,
            model_id=model_id,
            covariance_type=str(options.get("covariance_type", "cluster_entity")),
        )

    if model_type == "beta_regression":
        options = mixed_effects_options or {}
        return fit_beta_regression(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            add_intercept=bool(options.get("add_intercept", True)),
            maximum_iterations=int(options.get("max_iterations", options.get("maximum_iterations", 100))),
        )

    if model_type == "fractional_logit":
        options = mixed_effects_options or {}
        return fit_fractional_logit(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            covariance_type=str(options.get("covariance_type", "HC3")),
            add_intercept=bool(options.get("add_intercept", True)),
            maximum_iterations=int(options.get("max_iterations", options.get("maximum_iterations", 100))),
        )





    if model_type == "exponential_aft":
        options = mixed_effects_options or {}
        event_variable = str(options.get("event_variable", "")).strip()
        if not event_variable:
            raise ValueError("Exponential AFT regression requires event_variable.")
        return fit_exponential_aft(
            dataframe,
            duration_variable=dependent_variable,
            event_variable=event_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            add_intercept=bool(options.get("add_intercept", True)),
            maximum_iterations=int(options.get("max_iterations", options.get("maximum_iterations", 500))),
        )

    if model_type == "loglogistic_aft":
        options = mixed_effects_options or {}
        event_variable = str(options.get("event_variable", "")).strip()
        if not event_variable:
            raise ValueError("Log-logistic AFT regression requires event_variable.")
        return fit_loglogistic_aft(
            dataframe,
            duration_variable=dependent_variable,
            event_variable=event_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            add_intercept=bool(options.get("add_intercept", True)),
            maximum_iterations=int(options.get("max_iterations", options.get("maximum_iterations", 500))),
        )

    if model_type == "lognormal_aft":
        options = mixed_effects_options or {}
        event_variable = str(options.get("event_variable", "")).strip()
        if not event_variable:
            raise ValueError("Log-normal AFT regression requires event_variable.")
        return fit_lognormal_aft(
            dataframe,
            duration_variable=dependent_variable,
            event_variable=event_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            add_intercept=bool(options.get("add_intercept", True)),
            maximum_iterations=int(options.get("max_iterations", options.get("maximum_iterations", 500))),
        )

    if model_type == "weibull_aft":
        options = mixed_effects_options or {}
        event_variable = str(options.get("event_variable", "")).strip()
        if not event_variable:
            raise ValueError("Weibull AFT regression requires event_variable.")
        return fit_weibull_aft(
            dataframe,
            duration_variable=dependent_variable,
            event_variable=event_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            add_intercept=bool(options.get("add_intercept", True)),
            maximum_iterations=int(options.get("max_iterations", options.get("maximum_iterations", 500))),
        )

    if model_type == "cox_proportional_hazards":
        options = mixed_effects_options or {}
        event_variable = str(options.get("event_variable", "")).strip()
        if not event_variable:
            raise ValueError("Cox proportional hazards requires event_variable.")
        return fit_cox_proportional_hazards(
            dataframe,
            duration_variable=dependent_variable,
            event_variable=event_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            ties=str(options.get("ties", "breslow")),
            maximum_iterations=int(options.get("max_iterations", options.get("maximum_iterations", 100))),
        )

    if model_type == "quantile_regression":
        options = mixed_effects_options or {}
        return fit_quantile_regression(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            quantile=float(options.get("quantile", 0.5)),
            add_intercept=bool(options.get("add_intercept", True)),
            maximum_iterations=int(options.get("max_iterations", options.get("maximum_iterations", 1000))),
        )

    if model_type == "multinomial_logit":
        options = mixed_effects_options or {}
        return fit_multinomial_logit(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            covariance_type=str(options.get("covariance_type", "HC3")),
            add_intercept=bool(options.get("add_intercept", True)),
            maximum_iterations=int(options.get("max_iterations", options.get("maximum_iterations", 100))),
        )

    if model_type in {"gee_gaussian", "gee_logit", "gee_poisson"}:
        options = mixed_effects_options or {}
        gee_group = group_variable or str(options.get("group_variable", "")).strip()
        if not gee_group:
            raise ValueError("GEE requires group_variable.")
        return fit_gee(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            group_variable=gee_group,
            fixed_effects=fixed_effects,
            model_id=model_id,
            model_type=model_type,
            covariance_structure=str(options.get("covariance_structure", "exchangeable")),
            add_intercept=bool(options.get("add_intercept", True)),
            maximum_iterations=int(options.get("max_iterations", options.get("maximum_iterations", 100))),
        )

    if model_type == "mixed_negative_binomial_three_level":
        options = mixed_effects_options or {}
        level2_group = str(options.get("level2_group", "")).strip()
        level3_group = str(options.get("level3_group", "")).strip()
        if not level2_group or not level3_group:
            raise ValueError("three-level mixed negative binomial requires level2_group and level3_group.")
        return fit_mixed_negative_binomial_three_level(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            level2_group=level2_group,
            level3_group=level3_group,
            model_id=model_id,
            add_intercept=bool(options.get("add_intercept", True)),
            optimizer=str(options.get("optimizer", "BFGS")),
            max_iterations=int(options.get("max_iterations", 300)),
            quadrature_points=int(options.get("quadrature_points", 7)),
        )

    if model_type == "mixed_negative_binomial_random_slope":
        if group_variable is None or not group_variable.strip():
            raise ValueError("mixed negative binomial random slope requires group_variable.")
        options = mixed_effects_options or {}
        slope = str(options.get("random_slope_variable", "")).strip()
        raw_slopes = options.get("random_slope_variables")
        if not slope and isinstance(raw_slopes, (list, tuple)) and raw_slopes:
            slope = str(raw_slopes[0]).strip()
        if not slope:
            raise ValueError("mixed negative binomial random slope requires random_slope_variable.")
        return fit_mixed_negative_binomial_random_slope(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            group_variable=group_variable,
            random_slope_variable=slope,
            model_id=model_id,
            add_intercept=bool(options.get("add_intercept", True)),
            optimizer=str(options.get("optimizer", "BFGS")),
            max_iterations=int(options.get("max_iterations", 300)),
            quadrature_points=int(options.get("quadrature_points", 9)),
        )

    if model_type == "mixed_negative_binomial_random_intercept":
        if group_variable is None or not group_variable.strip():
            raise ValueError("혼합 음이항 Random Intercept 모형에는 그룹변수가 필요합니다.")
        options = mixed_effects_options or {}
        return fit_mixed_negative_binomial_random_intercept(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            group_variable=group_variable,
            model_id=model_id,
            add_intercept=bool(options.get("add_intercept", True)),
            optimizer=str(options.get("optimizer", "BFGS")),
            max_iterations=int(options.get("max_iterations", 300)),
            quadrature_points=int(options.get("quadrature_points", 15)),
        )

    if model_type == "mixed_poisson_three_level":
        options = mixed_effects_options or {}
        level2_group = str(options.get("level2_group", "")).strip()
        level3_group = str(options.get("level3_group", "")).strip()
        if not level2_group or not level3_group:
            raise ValueError("3수준 혼합 포아송에는 level2_group과 level3_group이 필요합니다.")
        return fit_mixed_poisson_three_level(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            level2_group=level2_group,
            level3_group=level3_group,
            model_id=model_id,
            add_intercept=bool(options.get("add_intercept", True)),
            optimizer=str(options.get("optimizer", "BFGS")),
            max_iterations=int(options.get("max_iterations", 300)),
            fe_prior_sd=float(options.get("fe_prior_sd", 2.0)),
            variance_prior_sd=float(options.get("variance_prior_sd", 1.0)),
        )

    if model_type == "mixed_poisson_random_slope":
        if group_variable is None or not group_variable.strip():
            raise ValueError("혼합 포아송 Random Slope 모형에는 그룹변수가 필요합니다.")
        options = mixed_effects_options or {}
        slope = str(options.get("random_slope_variable", "")).strip()
        raw_slopes = options.get("random_slope_variables")
        if not slope and isinstance(raw_slopes, (list, tuple)) and raw_slopes:
            slope = str(raw_slopes[0]).strip()
        if not slope:
            raise ValueError(
                "혼합 포아송 Random Slope 모형에는 random_slope_variable이 필요합니다."
            )
        return fit_mixed_poisson_random_slope(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            group_variable=group_variable,
            random_slope_variable=slope,
            model_id=model_id,
            add_intercept=bool(options.get("add_intercept", True)),
            optimizer=str(options.get("optimizer", "BFGS")),
            max_iterations=int(options.get("max_iterations", 200)),
            fe_prior_sd=float(options.get("fe_prior_sd", 2.0)),
            variance_prior_sd=float(options.get("variance_prior_sd", 1.0)),
        )

    if model_type == "mixed_poisson_random_intercept":
        if group_variable is None or not group_variable.strip():
            raise ValueError("혼합 포아송 Random Intercept 모형에는 그룹변수가 필요합니다.")
        options = mixed_effects_options or {}
        return fit_mixed_poisson_random_intercept(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            group_variable=group_variable,
            model_id=model_id,
            add_intercept=bool(options.get("add_intercept", True)),
            optimizer=str(options.get("optimizer", "BFGS")),
            max_iterations=int(options.get("max_iterations", 200)),
            fe_prior_sd=float(options.get("fe_prior_sd", 2.0)),
            variance_prior_sd=float(options.get("variance_prior_sd", 1.0)),
        )

    if model_type == "mixed_binary_logit_three_level":
        options = mixed_effects_options or {}
        level2_group = str(options.get("level2_group", "")).strip()
        level3_group = str(options.get("level3_group", "")).strip()
        if not level2_group or not level3_group:
            raise ValueError("3수준 혼합 이항 로짓에는 level2_group과 level3_group이 필요합니다.")
        return fit_mixed_binary_logit_three_level(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            level2_group=level2_group,
            level3_group=level3_group,
            model_id=model_id,
            add_intercept=bool(options.get("add_intercept", True)),
            optimizer=str(options.get("optimizer", "BFGS")),
            max_iterations=int(options.get("max_iterations", 300)),
            fe_prior_sd=float(options.get("fe_prior_sd", 2.0)),
            variance_prior_sd=float(options.get("variance_prior_sd", 1.0)),
        )

    if model_type == "mixed_binary_logit_random_slope":
        if group_variable is None or not group_variable.strip():
            raise ValueError("혼합 이항 로짓 Random Slope 모형에는 그룹변수가 필요합니다.")
        options = mixed_effects_options or {}
        slope = str(options.get("random_slope_variable", "")).strip()
        raw_slopes = options.get("random_slope_variables")
        if not slope and isinstance(raw_slopes, (list, tuple)) and raw_slopes:
            slope = str(raw_slopes[0]).strip()
        if not slope:
            raise ValueError(
                "혼합 이항 로짓 Random Slope 모형에는 random_slope_variable이 필요합니다."
            )
        return fit_mixed_binary_logit_random_slope(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            group_variable=group_variable,
            random_slope_variable=slope,
            model_id=model_id,
            add_intercept=bool(options.get("add_intercept", True)),
            optimizer=str(options.get("optimizer", "BFGS")),
            max_iterations=int(options.get("max_iterations", 200)),
            fe_prior_sd=float(options.get("fe_prior_sd", 2.0)),
            variance_prior_sd=float(options.get("variance_prior_sd", 1.0)),
        )

    if model_type == "mixed_binary_logit_random_intercept":
        if group_variable is None or not group_variable.strip():
            raise ValueError("혼합 이항 로짓 Random Intercept 모형에는 그룹변수가 필요합니다.")
        options = mixed_effects_options or {}
        return fit_mixed_binary_logit_random_intercept(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            group_variable=group_variable,
            model_id=model_id,
            add_intercept=bool(options.get("add_intercept", True)),
            optimizer=str(options.get("optimizer", "BFGS")),
            max_iterations=int(options.get("max_iterations", 200)),
            fe_prior_sd=float(options.get("fe_prior_sd", 2.0)),
            variance_prior_sd=float(options.get("variance_prior_sd", 1.0)),
        )

    if model_type == "mixed_three_level":
        options = mixed_effects_options or {}
        level2_group = str(options.get("level2_group", "")).strip()
        level3_group = str(options.get("level3_group", "")).strip()
        if not level2_group or not level3_group:
            raise ValueError("3수준 혼합효과 모형에는 level2_group과 level3_group이 필요합니다.")
        return fit_three_level_mixed_effects(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            level2_group=level2_group,
            level3_group=level3_group,
            level2_random_slope_variables=options.get("level2_random_slope_variables"),
            level3_random_slope_variables=options.get("level3_random_slope_variables"),
            model_id=model_id,
            add_intercept=bool(options.get("add_intercept", True)),
            reml=bool(options.get("reml", False)),
            method=str(options.get("optimizer", "lbfgs")),
            max_iterations=int(options.get("max_iterations", 300)),
        )

    if model_type in {"mixed_random_intercept", "mixed_random_slope", "mixed_three_level"}:
        if group_variable is None or not group_variable.strip():
            raise ValueError("Random Intercept 모형에는 그룹변수가 필요합니다.")

        options = mixed_effects_options or {}

        fit_function = fit_random_intercept
        extra = {}
        if model_type == "mixed_random_slope":
            raw_slopes = options.get("random_slope_variables")
            slopes = (
                [str(v).strip() for v in raw_slopes]
                if isinstance(raw_slopes, (list, tuple))
                else []
            )
            slopes = [v for v in slopes if v]
            if not slopes:
                slope = str(options.get("random_slope_variable", "")).strip()
                slopes = [slope] if slope else []
            if not slopes:
                raise ValueError(
                    "Random Slope 모형에는 random_slope_variable 또는 random_slope_variables가 필요합니다."
                )
            fit_function = fit_multiple_random_slopes if len(slopes) > 1 else fit_random_slope
            if len(slopes) > 1:
                extra["random_slope_variables"] = slopes
            else:
                extra["random_slope_variable"] = slopes[0]
            predictor = str(options.get("cross_level_predictor", "")).strip()
            moderator = str(options.get("cross_level_moderator", "")).strip()
            if predictor or moderator:
                extra.update(
                    {
                        "cross_level_predictor": predictor or None,
                        "cross_level_moderator": moderator or None,
                        "level1_centering": str(options.get("level1_centering", "none")),
                        "level2_centering": str(options.get("level2_centering", "none")),
                        "simple_slope_values": options.get("simple_slope_values"),
                        "johnson_neyman": bool(options.get("johnson_neyman", False)),
                    }
                )

        return fit_function(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            group_variable=group_variable,
            model_id=model_id,
            add_intercept=bool(options.get("add_intercept", True)),
            reml=bool(options.get("reml", False)),
            method=str(options.get("optimizer", "lbfgs")),
            max_iterations=int(options.get("max_iterations", 200)),
            random_effect_covariance=str(options.get("random_effect_covariance", "correlated")),
            **extra,
        )

    if measurement_level == "continuous":
        return fit_ols(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            covariance_type="HC3",
        )
    if measurement_level == "binary":
        return fit_binary_logit(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            covariance_type="HC3",
        )
    if measurement_level == "proportion":
        return fit_fractional_logit(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            covariance_type="HC3",
        )
    if measurement_level == "nominal":
        return fit_multinomial_logit(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            covariance_type="HC3",
        )
    if measurement_level in {"ordinal", "scale_item"}:
        return fit_ordered_logit(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
        )
    if measurement_level == "count":
        return fit_count_regression(
            dataframe,
            dependent_variable=dependent_variable,
            independent_variables=independent_variables,
            fixed_effects=fixed_effects,
            model_id=model_id,
            covariance_type="HC3",
        )
    raise ValueError(f"지원하지 않는 종속변수 측정수준입니다: {measurement_level}")
