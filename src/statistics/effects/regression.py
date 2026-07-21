"""회귀모형별 효과크기 및 평균한계효과 계산."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.statistics.regression.base import RegressionResult


@dataclass(slots=True)
class EffectSizeResult:
    """변수별 효과크기 결과."""

    term: str
    effect_type: str
    estimate: float | None
    standard_error: float | None
    statistic: float | None
    p_value: float | None
    confidence_interval_lower: float | None
    confidence_interval_upper: float | None
    magnitude: str | None
    interpretation: str


@dataclass(slots=True)
class EffectSizeReport:
    """회귀 효과크기 전체 결과."""

    model_id: str
    model_type: str
    effects: list[EffectSizeResult]
    model_effects: dict[str, float | None]
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _magnitude_from_partial_r_squared(value: float) -> str:
    if value < 0.01:
        return "negligible"
    if value < 0.09:
        return "small"
    if value < 0.25:
        return "medium"
    return "large"


def _magnitude_from_cohen_f_squared(value: float) -> str:
    if value < 0.02:
        return "negligible"
    if value < 0.15:
        return "small"
    if value < 0.35:
        return "medium"
    return "large"


def _build_ols_effects(
    result: RegressionResult,
) -> EffectSizeReport:
    fitted = result.raw_result

    if fitted is None:
        raise ValueError("원본 OLS 결과 객체가 없습니다.")

    endog = np.asarray(fitted.model.endog, dtype=float)
    exog = np.asarray(fitted.model.exog, dtype=float)
    exog_names = list(fitted.model.exog_names)
    outcome_sd = float(np.std(endog, ddof=1))
    residual_df = float(fitted.df_resid)

    if np.isclose(outcome_sd, 0.0):
        raise ValueError("종속변수 표준편차가 0이므로 표준화 계수를 계산할 수 없습니다.")

    effects: list[EffectSizeResult] = []
    coefficient_lookup = {coefficient.term: coefficient for coefficient in result.coefficients}

    for index, term in enumerate(exog_names):
        if str(term).lower() in {"const", "intercept"}:
            continue

        coefficient = coefficient_lookup[str(term)]
        predictor_sd = float(np.std(exog[:, index], ddof=1))
        standardized_beta = coefficient.estimate * predictor_sd / outcome_sd

        t_value = float(coefficient.statistic)
        partial_r_squared = t_value**2 / (t_value**2 + residual_df) if residual_df > 0 else np.nan
        partial_f_squared = (
            partial_r_squared / (1 - partial_r_squared) if partial_r_squared < 1 else np.inf
        )

        effects.extend(
            [
                EffectSizeResult(
                    term=str(term),
                    effect_type="standardized_beta",
                    estimate=float(standardized_beta),
                    standard_error=None,
                    statistic=t_value,
                    p_value=float(coefficient.p_value),
                    confidence_interval_lower=None,
                    confidence_interval_upper=None,
                    magnitude=None,
                    interpretation=(
                        "독립변수가 1표준편차 증가할 때 종속변수가 "
                        f"{standardized_beta:.3f}표준편차 변화합니다."
                    ),
                ),
                EffectSizeResult(
                    term=str(term),
                    effect_type="partial_r_squared",
                    estimate=float(partial_r_squared),
                    standard_error=None,
                    statistic=t_value,
                    p_value=float(coefficient.p_value),
                    confidence_interval_lower=None,
                    confidence_interval_upper=None,
                    magnitude=_magnitude_from_partial_r_squared(float(partial_r_squared)),
                    interpretation=("다른 변수를 통제한 뒤 해당 변수의 고유 설명력입니다."),
                ),
                EffectSizeResult(
                    term=str(term),
                    effect_type="partial_cohen_f_squared",
                    estimate=float(partial_f_squared),
                    standard_error=None,
                    statistic=t_value,
                    p_value=float(coefficient.p_value),
                    confidence_interval_lower=None,
                    confidence_interval_upper=None,
                    magnitude=_magnitude_from_cohen_f_squared(float(partial_f_squared)),
                    interpretation=("부분 R²를 Cohen's f²로 변환한 효과크기입니다."),
                ),
            ]
        )

    r_squared = float(result.fit_statistics["r_squared"])
    model_f_squared = r_squared / (1 - r_squared) if r_squared < 1 else np.inf

    return EffectSizeReport(
        model_id=result.model_id,
        model_type=result.model_type,
        effects=effects,
        model_effects={
            "r_squared": r_squared,
            "adjusted_r_squared": float(result.fit_statistics["adjusted_r_squared"]),
            "model_cohen_f_squared": model_f_squared,
        },
        metadata={
            "sample_size": result.sample_size,
            "residual_degrees_of_freedom": residual_df,
        },
    )


def _build_beta_regression_effects(result: RegressionResult) -> EffectSizeReport:
    effects: list[EffectSizeResult] = []
    for coefficient in result.coefficients:
        if coefficient.term.lower() in {"const", "intercept", "precision"}:
            continue
        effects.append(
            EffectSizeResult(
                term=coefficient.term,
                effect_type="mean_odds_ratio",
                estimate=coefficient.exponentiated_estimate,
                standard_error=None,
                statistic=coefficient.statistic,
                p_value=coefficient.p_value,
                confidence_interval_lower=float(np.exp(coefficient.confidence_interval_lower)),
                confidence_interval_upper=float(np.exp(coefficient.confidence_interval_upper)),
                magnitude=None,
                interpretation="Odds-scale effect on the beta-regression conditional mean.",
            )
        )
    return EffectSizeReport(
        model_id=result.model_id,
        model_type=result.model_type,
        effects=effects,
        model_effects={
            "pseudo_r_squared": result.fit_statistics.get("pseudo_r_squared"),
            "precision": result.fit_statistics.get("precision"),
            "mean_absolute_error": result.fit_statistics.get("mean_absolute_error"),
        },
        metadata={"sample_size": result.sample_size},
    )


def _build_fractional_logit_effects(result: RegressionResult) -> EffectSizeReport:
    effects: list[EffectSizeResult] = []
    warnings: list[str] = []
    fitted = result.raw_result
    for coefficient in result.coefficients:
        if coefficient.term.lower() in {"const", "intercept"}:
            continue
        effects.append(
            EffectSizeResult(
                term=coefficient.term,
                effect_type="fractional_odds_ratio",
                estimate=coefficient.exponentiated_estimate,
                standard_error=None,
                statistic=coefficient.statistic,
                p_value=coefficient.p_value,
                confidence_interval_lower=float(np.exp(coefficient.confidence_interval_lower)),
                confidence_interval_upper=float(np.exp(coefficient.confidence_interval_upper)),
                magnitude=None,
                interpretation="Odds-scale effect for a fractional logit mean model.",
            )
        )
    if fitted is not None:
        try:
            marginal = fitted.get_margeff(at="overall", method="dydx").summary_frame()
            for term in marginal.index:
                effects.append(
                    EffectSizeResult(
                        term=str(term),
                        effect_type="average_marginal_effect",
                        estimate=float(marginal.loc[term, "dy/dx"]),
                        standard_error=float(marginal.loc[term, "Std. Err."]),
                        statistic=float(marginal.loc[term, "z"]),
                        p_value=float(marginal.loc[term, "Pr(>|z|)"]),
                        confidence_interval_lower=float(marginal.loc[term, "Conf. Int. Low"]),
                        confidence_interval_upper=float(marginal.loc[term, "Cont. Int. Hi."] if "Cont. Int. Hi." in marginal.columns else marginal.loc[term, "Conf. Int. Hi."]),
                        magnitude=None,
                        interpretation="Average marginal effect on the expected proportion.",
                    )
                )
        except (AttributeError, KeyError, ValueError, np.linalg.LinAlgError) as error:
            warnings.append(f"Average marginal effects could not be computed: {error}")
    return EffectSizeReport(
        model_id=result.model_id,
        model_type=result.model_type,
        effects=effects,
        model_effects={
            "pseudo_r_squared_deviance": result.fit_statistics.get("pseudo_r_squared_deviance"),
            "dispersion_ratio": result.fit_statistics.get("dispersion_ratio"),
            "mean_absolute_error": result.fit_statistics.get("mean_absolute_error"),
        },
        warnings=warnings,
        metadata={"sample_size": result.sample_size},
    )


def _build_cox_effects(result: RegressionResult) -> EffectSizeReport:
    effects: list[EffectSizeResult] = []
    for coefficient in result.coefficients:
        if coefficient.term.lower() in {"const", "intercept"}:
            continue
        effects.append(
            EffectSizeResult(
                term=coefficient.term,
                effect_type="hazard_ratio",
                estimate=coefficient.exponentiated_estimate,
                standard_error=None,
                statistic=coefficient.statistic,
                p_value=coefficient.p_value,
                confidence_interval_lower=float(np.exp(coefficient.confidence_interval_lower)),
                confidence_interval_upper=float(np.exp(coefficient.confidence_interval_upper)),
                magnitude=None,
                interpretation="Hazard ratio from a Cox proportional hazards model.",
            )
        )
    return EffectSizeReport(
        model_id=result.model_id,
        model_type=result.model_type,
        effects=effects,
        model_effects={
            "event_count": result.fit_statistics.get("event_count"),
            "censored_count": result.fit_statistics.get("censored_count"),
            "events_per_parameter": result.fit_statistics.get("events_per_parameter"),
        },
        metadata={
            "sample_size": result.sample_size,
            "duration_variable": result.metadata.get("duration_variable"),
            "event_variable": result.metadata.get("event_variable"),
        },
    )


def _build_quantile_effects(result: RegressionResult) -> EffectSizeReport:
    fitted = result.raw_result
    if fitted is None:
        raise ValueError("A fitted quantile regression result is required.")

    endog = np.asarray(fitted.model.endog, dtype=float)
    exog = np.asarray(fitted.model.exog, dtype=float)
    exog_names = [str(name) for name in fitted.model.exog_names]
    outcome_sd = float(np.std(endog, ddof=1))
    effects: list[EffectSizeResult] = []
    warnings: list[str] = []
    if not np.isfinite(outcome_sd) or np.isclose(outcome_sd, 0.0):
        warnings.append("Quantile standardized effects could not be computed because outcome SD is zero.")
        outcome_sd = np.nan
    coefficient_lookup = {coefficient.term: coefficient for coefficient in result.coefficients}
    for index, term in enumerate(exog_names):
        if term.lower() in {"const", "intercept"}:
            continue
        coefficient = coefficient_lookup.get(term)
        if coefficient is None or not np.isfinite(outcome_sd):
            continue
        predictor_sd = float(np.std(exog[:, index], ddof=1))
        estimate = coefficient.estimate * predictor_sd / outcome_sd
        effects.append(
            EffectSizeResult(
                term=term,
                effect_type="standardized_quantile_beta",
                estimate=float(estimate),
                standard_error=None,
                statistic=coefficient.statistic,
                p_value=coefficient.p_value,
                confidence_interval_lower=None,
                confidence_interval_upper=None,
                magnitude=None,
                interpretation="Standardized coefficient for the modeled conditional quantile.",
            )
        )

    return EffectSizeReport(
        model_id=result.model_id,
        model_type=result.model_type,
        effects=effects,
        model_effects={
            "quantile": result.fit_statistics.get("quantile"),
            "pseudo_r_squared": result.fit_statistics.get("pseudo_r_squared"),
            "pinball_loss": result.fit_statistics.get("pinball_loss"),
        },
        warnings=warnings,
        metadata={"sample_size": result.sample_size},
    )


def _build_heckman_effects(result: RegressionResult) -> EffectSizeReport:
    fitted = result.raw_result
    if fitted is None:
        raise ValueError("A fitted Heckman result is required.")
    endog = np.asarray(fitted.outcome_result.model.endog, dtype=float)
    exog = np.asarray(fitted.outcome_result.model.exog, dtype=float)
    exog_names = [str(name) for name in fitted.outcome_result.model.exog_names]
    outcome_sd = float(np.std(endog, ddof=1)) if endog.size > 1 else np.nan
    effects: list[EffectSizeResult] = []
    warnings: list[str] = []
    coefficient_lookup = {coefficient.term: coefficient for coefficient in result.coefficients}
    if not np.isfinite(outcome_sd) or np.isclose(outcome_sd, 0.0):
        warnings.append("Heckman standardized effects could not be computed because outcome SD is zero.")
    for index, term in enumerate(exog_names):
        if term.lower() in {"const", "intercept", "inverse_mills_ratio"}:
            continue
        coefficient = coefficient_lookup.get(term)
        if coefficient is None:
            continue
        predictor_sd = float(np.std(exog[:, index], ddof=1))
        estimate = None
        if np.isfinite(outcome_sd) and not np.isclose(outcome_sd, 0.0):
            estimate = float(coefficient.estimate * predictor_sd / outcome_sd)
        effects.append(
            EffectSizeResult(
                term=term,
                effect_type="heckman_standardized_beta",
                estimate=estimate,
                standard_error=None,
                statistic=coefficient.statistic,
                p_value=coefficient.p_value,
                confidence_interval_lower=None,
                confidence_interval_upper=None,
                magnitude=None,
                interpretation="Standardized outcome-equation coefficient from Heckman two-step estimation.",
            )
        )
    return EffectSizeReport(
        model_id=result.model_id,
        model_type=result.model_type,
        effects=effects,
        model_effects={
            "outcome_r_squared": result.fit_statistics.get("outcome_r_squared"),
            "inverse_mills_coefficient": result.fit_statistics.get("inverse_mills_coefficient"),
            "inverse_mills_p_value": result.fit_statistics.get("inverse_mills_p_value"),
            "selection_rate": result.fit_statistics.get("selection_rate"),
        },
        warnings=warnings,
        metadata={
            "sample_size": result.sample_size,
            "selection_variable": result.metadata.get("selection_variable"),
            "exclusion_restrictions": result.metadata.get("exclusion_restrictions"),
        },
    )


def _build_iv_2sls_effects(result: RegressionResult) -> EffectSizeReport:
    fitted = result.raw_result
    if fitted is None:
        raise ValueError("A fitted IV 2SLS result is required.")
    endog = np.asarray(fitted.model.endog, dtype=float)
    exog = np.asarray(fitted.model.exog, dtype=float)
    exog_names = [str(name) for name in fitted.model.exog_names]
    outcome_sd = float(np.std(endog, ddof=1)) if endog.size > 1 else np.nan
    effects: list[EffectSizeResult] = []
    warnings: list[str] = []
    if not np.isfinite(outcome_sd) or np.isclose(outcome_sd, 0.0):
        warnings.append("IV standardized effects could not be computed because outcome SD is zero.")
    coefficient_lookup = {coefficient.term: coefficient for coefficient in result.coefficients}
    for index, term in enumerate(exog_names):
        if term.lower() in {"const", "intercept"}:
            continue
        coefficient = coefficient_lookup.get(term)
        if coefficient is None:
            continue
        predictor_sd = float(np.std(exog[:, index], ddof=1))
        estimate = None
        if np.isfinite(outcome_sd) and not np.isclose(outcome_sd, 0.0):
            estimate = float(coefficient.estimate * predictor_sd / outcome_sd)
        effects.append(
            EffectSizeResult(
                term=term,
                effect_type="iv_standardized_beta",
                estimate=estimate,
                standard_error=None,
                statistic=coefficient.statistic,
                p_value=coefficient.p_value,
                confidence_interval_lower=None,
                confidence_interval_upper=None,
                magnitude=None,
                interpretation="Standardized coefficient from two-stage least squares estimation.",
            )
        )
    return EffectSizeReport(
        model_id=result.model_id,
        model_type=result.model_type,
        effects=effects,
        model_effects={
            "r_squared": result.fit_statistics.get("r_squared"),
            "minimum_first_stage_f_statistic": result.fit_statistics.get("minimum_first_stage_f_statistic"),
            "instrument_count": result.fit_statistics.get("instrument_count"),
            "endogenous_variable_count": result.fit_statistics.get("endogenous_variable_count"),
        },
        warnings=warnings,
        metadata={
            "sample_size": result.sample_size,
            "endogenous_variables": result.metadata.get("endogenous_variables"),
            "instrument_variables": result.metadata.get("instrument_variables"),
        },
    )


def _build_inverse_gaussian_effects(result: RegressionResult) -> EffectSizeReport:
    effects: list[EffectSizeResult] = []
    for coefficient in result.coefficients:
        if coefficient.term.lower() in {"const", "intercept"}:
            continue
        effects.append(
            EffectSizeResult(
                term=coefficient.term,
                effect_type="mean_ratio",
                estimate=coefficient.exponentiated_estimate,
                standard_error=None,
                statistic=coefficient.statistic,
                p_value=coefficient.p_value,
                confidence_interval_lower=float(np.exp(coefficient.confidence_interval_lower)),
                confidence_interval_upper=float(np.exp(coefficient.confidence_interval_upper)),
                magnitude=None,
                interpretation="Multiplicative mean ratio from an Inverse Gaussian log-link model.",
            )
        )
    return EffectSizeReport(
        model_id=result.model_id,
        model_type=result.model_type,
        effects=effects,
        model_effects={
            "pseudo_r_squared_deviance": result.fit_statistics.get("pseudo_r_squared_deviance"),
            "dispersion_ratio": result.fit_statistics.get("dispersion_ratio"),
            "root_mean_squared_error": result.fit_statistics.get("root_mean_squared_error"),
        },
        metadata={
            "sample_size": result.sample_size,
            "link": result.metadata.get("link"),
            "family": result.metadata.get("family"),
        },
    )


def _build_gamma_effects(result: RegressionResult) -> EffectSizeReport:
    effects: list[EffectSizeResult] = []
    for coefficient in result.coefficients:
        if coefficient.term.lower() in {"const", "intercept"}:
            continue
        effects.append(
            EffectSizeResult(
                term=coefficient.term,
                effect_type="mean_ratio",
                estimate=coefficient.exponentiated_estimate,
                standard_error=None,
                statistic=coefficient.statistic,
                p_value=coefficient.p_value,
                confidence_interval_lower=float(np.exp(coefficient.confidence_interval_lower)),
                confidence_interval_upper=float(np.exp(coefficient.confidence_interval_upper)),
                magnitude=None,
                interpretation="Multiplicative mean ratio from a Gamma log-link model.",
            )
        )
    return EffectSizeReport(
        model_id=result.model_id,
        model_type=result.model_type,
        effects=effects,
        model_effects={
            "pseudo_r_squared_deviance": result.fit_statistics.get("pseudo_r_squared_deviance"),
            "dispersion_ratio": result.fit_statistics.get("dispersion_ratio"),
            "root_mean_squared_error": result.fit_statistics.get("root_mean_squared_error"),
        },
        metadata={
            "sample_size": result.sample_size,
            "link": result.metadata.get("link"),
            "family": result.metadata.get("family"),
        },
    )


def _build_regularized_effects(result: RegressionResult) -> EffectSizeReport:
    fitted = result.raw_result
    if fitted is None:
        raise ValueError("A fitted regularized regression result is required.")
    endog = np.asarray(fitted.model.endog, dtype=float)
    standardized = result.metadata.get("standardized_coefficients", {})
    outcome_sd = float(np.std(endog, ddof=1)) if endog.size > 1 else np.nan
    effects: list[EffectSizeResult] = []
    warnings: list[str] = []
    if not np.isfinite(outcome_sd) or np.isclose(outcome_sd, 0.0):
        warnings.append("Regularized standardized effects could not be computed because outcome SD is zero.")
    for coefficient in result.coefficients:
        if coefficient.term.lower() in {"const", "intercept"}:
            continue
        estimate = None
        if coefficient.term in standardized and np.isfinite(outcome_sd) and not np.isclose(outcome_sd, 0.0):
            estimate = float(standardized[coefficient.term] / outcome_sd)
        effects.append(
            EffectSizeResult(
                term=coefficient.term,
                effect_type="regularized_standardized_beta",
                estimate=estimate,
                standard_error=None,
                statistic=None,
                p_value=None,
                confidence_interval_lower=None,
                confidence_interval_upper=None,
                magnitude=None,
                interpretation="Standardized coefficient from penalized least-squares estimation.",
            )
        )

    return EffectSizeReport(
        model_id=result.model_id,
        model_type=result.model_type,
        effects=effects,
        model_effects={
            "penalty": result.fit_statistics.get("penalty"),
            "alpha": result.fit_statistics.get("alpha"),
            "l1_ratio": result.fit_statistics.get("l1_ratio"),
            "pseudo_r_squared": result.fit_statistics.get("pseudo_r_squared"),
            "selected_coefficient_count": result.fit_statistics.get("selected_coefficient_count"),
            "zero_coefficient_count": result.fit_statistics.get("zero_coefficient_count"),
        },
        warnings=warnings,
        metadata={
            "sample_size": result.sample_size,
            "selected_terms": result.metadata.get("selected_terms"),
            "zero_terms": result.metadata.get("zero_terms"),
        },
    )


def _build_robust_effects(result: RegressionResult) -> EffectSizeReport:
    fitted = result.raw_result
    if fitted is None:
        raise ValueError("A fitted robust regression result is required.")
    endog = np.asarray(fitted.model.endog, dtype=float)
    exog = np.asarray(fitted.model.exog, dtype=float)
    exog_names = [str(name) for name in fitted.model.exog_names]
    outcome_sd = float(np.std(endog, ddof=1)) if endog.size > 1 else np.nan
    effects: list[EffectSizeResult] = []
    warnings: list[str] = []
    if not np.isfinite(outcome_sd) or np.isclose(outcome_sd, 0.0):
        warnings.append("Robust standardized effects could not be computed because outcome SD is zero.")
    coefficient_lookup = {coefficient.term: coefficient for coefficient in result.coefficients}
    for index, term in enumerate(exog_names):
        if term.lower() in {"const", "intercept"}:
            continue
        coefficient = coefficient_lookup.get(term)
        if coefficient is None:
            continue
        predictor_sd = float(np.std(exog[:, index], ddof=1))
        estimate = None
        if np.isfinite(outcome_sd) and not np.isclose(outcome_sd, 0.0):
            estimate = float(coefficient.estimate * predictor_sd / outcome_sd)
        effects.append(
            EffectSizeResult(
                term=term,
                effect_type="robust_standardized_beta",
                estimate=estimate,
                standard_error=None,
                statistic=coefficient.statistic,
                p_value=coefficient.p_value,
                confidence_interval_lower=None,
                confidence_interval_upper=None,
                magnitude=None,
                interpretation="Standardized coefficient from robust M-estimation.",
            )
        )

    return EffectSizeReport(
        model_id=result.model_id,
        model_type=result.model_type,
        effects=effects,
        model_effects={
            "pseudo_r_squared": result.fit_statistics.get("pseudo_r_squared"),
            "scale": result.fit_statistics.get("scale"),
            "downweighted_count": result.fit_statistics.get("downweighted_count"),
            "heavily_downweighted_count": result.fit_statistics.get("heavily_downweighted_count"),
        },
        warnings=warnings,
        metadata={
            "sample_size": result.sample_size,
            "norm": result.metadata.get("norm"),
        },
    )


def _build_tobit_effects(result: RegressionResult) -> EffectSizeReport:
    fitted = result.raw_result
    if fitted is None:
        raise ValueError("A fitted Tobit result is required.")
    endog = np.asarray(fitted.model.endog, dtype=float)
    exog = np.asarray(fitted.model.exog, dtype=float)
    exog_names = [str(name) for name in fitted.model.exog_names]
    outcome_sd = float(np.std(endog, ddof=1)) if endog.size > 1 else np.nan
    effects: list[EffectSizeResult] = []
    warnings: list[str] = []
    if not np.isfinite(outcome_sd) or np.isclose(outcome_sd, 0.0):
        warnings.append("Tobit standardized effects could not be computed because outcome SD is zero.")
    coefficient_lookup = {coefficient.term: coefficient for coefficient in result.coefficients}
    uncensored_probability = 1.0 - float(result.fit_statistics.get("censoring_rate", 0.0) or 0.0)
    for index, term in enumerate(exog_names):
        if term.lower() in {"const", "intercept"}:
            continue
        coefficient = coefficient_lookup.get(term)
        if coefficient is None:
            continue
        predictor_sd = float(np.std(exog[:, index], ddof=1))
        standardized = None
        if np.isfinite(outcome_sd) and not np.isclose(outcome_sd, 0.0):
            standardized = float(coefficient.estimate * predictor_sd / outcome_sd)
        effects.append(
            EffectSizeResult(
                term=term,
                effect_type="latent_standardized_beta",
                estimate=standardized,
                standard_error=None,
                statistic=coefficient.statistic,
                p_value=coefficient.p_value,
                confidence_interval_lower=None,
                confidence_interval_upper=None,
                magnitude=None,
                interpretation="Standardized coefficient on the Tobit latent outcome scale.",
            )
        )
        effects.append(
            EffectSizeResult(
                term=term,
                effect_type="observed_scale_marginal_effect",
                estimate=float(coefficient.estimate * uncensored_probability),
                standard_error=None,
                statistic=coefficient.statistic,
                p_value=coefficient.p_value,
                confidence_interval_lower=None,
                confidence_interval_upper=None,
                magnitude=None,
                interpretation="Approximate marginal effect on the observed censored outcome scale.",
            )
        )

    return EffectSizeReport(
        model_id=result.model_id,
        model_type=result.model_type,
        effects=effects,
        model_effects={
            "pseudo_r_squared": result.fit_statistics.get("pseudo_r_squared"),
            "censoring_rate": result.fit_statistics.get("censoring_rate"),
            "sigma": result.fit_statistics.get("sigma"),
        },
        warnings=warnings,
        metadata={
            "sample_size": result.sample_size,
            "lower_limit": result.metadata.get("lower_limit"),
            "upper_limit": result.metadata.get("upper_limit"),
        },
    )


def _build_panel_fixed_effects(result: RegressionResult) -> EffectSizeReport:
    within_outcome = np.asarray(result.metadata.get("within_outcome", []), dtype=float)
    within_predictors = np.asarray(result.metadata.get("within_predictors", []), dtype=float)
    predictor_names = [str(name) for name in result.metadata.get("within_predictor_names", [])]
    outcome_sd = float(np.std(within_outcome, ddof=1)) if within_outcome.size > 1 else np.nan
    effects: list[EffectSizeResult] = []
    warnings: list[str] = []
    if not np.isfinite(outcome_sd) or np.isclose(outcome_sd, 0.0):
        warnings.append("Panel standardized effects could not be computed because within-outcome SD is zero.")
    coefficient_lookup = {coefficient.term: coefficient for coefficient in result.coefficients}
    for index, term in enumerate(predictor_names):
        coefficient = coefficient_lookup.get(term)
        if coefficient is None or within_predictors.ndim != 2:
            continue
        predictor_sd = float(np.std(within_predictors[:, index], ddof=1))
        if not np.isfinite(outcome_sd) or np.isclose(outcome_sd, 0.0):
            estimate = None
        else:
            estimate = float(coefficient.estimate * predictor_sd / outcome_sd)
        effects.append(
            EffectSizeResult(
                term=term,
                effect_type="within_standardized_beta",
                estimate=estimate,
                standard_error=None,
                statistic=coefficient.statistic,
                p_value=coefficient.p_value,
                confidence_interval_lower=None,
                confidence_interval_upper=None,
                magnitude=None,
                interpretation="Standardized within-panel coefficient after absorbing fixed effects.",
            )
        )

    return EffectSizeReport(
        model_id=result.model_id,
        model_type=result.model_type,
        effects=effects,
        model_effects={
            "within_r_squared": result.fit_statistics.get("within_r_squared"),
            "adjusted_within_r_squared": result.fit_statistics.get("adjusted_within_r_squared"),
            "entity_count": result.fit_statistics.get("entity_count"),
            "time_period_count": result.fit_statistics.get("time_period_count"),
        },
        warnings=warnings,
        metadata={
            "sample_size": result.sample_size,
            "entity_variable": result.metadata.get("entity_variable"),
            "time_variable": result.metadata.get("time_variable"),
            "absorbed_effects": result.metadata.get("absorbed_effects"),
        },
    )


def _build_binary_logit_effects(
    result: RegressionResult,
) -> EffectSizeReport:
    fitted = result.raw_result

    if fitted is None:
        raise ValueError("원본 이항 로짓 결과 객체가 없습니다.")

    effects: list[EffectSizeResult] = []
    coefficient_lookup = {coefficient.term: coefficient for coefficient in result.coefficients}

    for term, coefficient in coefficient_lookup.items():
        if term.lower() in {"const", "intercept"}:
            continue

        odds_ratio = coefficient.exponentiated_estimate

        effects.append(
            EffectSizeResult(
                term=term,
                effect_type="odds_ratio",
                estimate=odds_ratio,
                standard_error=None,
                statistic=coefficient.statistic,
                p_value=coefficient.p_value,
                confidence_interval_lower=float(np.exp(coefficient.confidence_interval_lower)),
                confidence_interval_upper=float(np.exp(coefficient.confidence_interval_upper)),
                magnitude=None,
                interpretation=("독립변수가 1단위 증가할 때 사건 발생 오즈의 배수입니다."),
            )
        )

    warnings: list[str] = []

    try:
        marginal_effects = fitted.get_margeff(
            at="overall",
            method="dydx",
        )
        frame = marginal_effects.summary_frame()

        for term in frame.index:
            effects.append(
                EffectSizeResult(
                    term=str(term),
                    effect_type="average_marginal_effect",
                    estimate=float(frame.loc[term, "dy/dx"]),
                    standard_error=float(frame.loc[term, "Std. Err."]),
                    statistic=float(frame.loc[term, "z"]),
                    p_value=float(frame.loc[term, "Pr(>|z|)"]),
                    confidence_interval_lower=float(frame.loc[term, "Conf. Int. Low"]),
                    confidence_interval_upper=float(
                        frame.loc[term, "Cont. Int. Hi."]
                        if "Cont. Int. Hi." in frame.columns
                        else frame.loc[term, "Conf. Int. Hi."]
                    ),
                    magnitude=None,
                    interpretation=("표본 전체에서 계산한 평균한계효과입니다."),
                )
            )
    except (
        AttributeError,
        KeyError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        warnings.append(f"평균한계효과를 계산하지 못했습니다: {error}")

    return EffectSizeReport(
        model_id=result.model_id,
        model_type=result.model_type,
        effects=effects,
        model_effects={
            "mcfadden_pseudo_r_squared": result.fit_statistics.get("pseudo_r_squared_mcfadden"),
        },
        warnings=warnings,
        metadata={
            "sample_size": result.sample_size,
        },
    )


def _build_binary_probit_effects(result: RegressionResult) -> EffectSizeReport:
    fitted = result.raw_result
    if fitted is None:
        raise ValueError("A fitted binary probit result is required.")

    effects: list[EffectSizeResult] = []
    for coefficient in result.coefficients:
        if coefficient.term.lower() in {"const", "intercept"}:
            continue
        effects.append(
            EffectSizeResult(
                term=coefficient.term,
                effect_type="probit_latent_coefficient",
                estimate=coefficient.estimate,
                standard_error=coefficient.standard_error,
                statistic=coefficient.statistic,
                p_value=coefficient.p_value,
                confidence_interval_lower=coefficient.confidence_interval_lower,
                confidence_interval_upper=coefficient.confidence_interval_upper,
                magnitude=None,
                interpretation="Latent-index coefficient from a binary probit model.",
            )
        )

    warnings: list[str] = []
    try:
        marginal_effects = fitted.get_margeff(at="overall", method="dydx")
        frame = marginal_effects.summary_frame()
        for term in frame.index:
            effects.append(
                EffectSizeResult(
                    term=str(term),
                    effect_type="average_marginal_effect",
                    estimate=float(frame.loc[term, "dy/dx"]),
                    standard_error=float(frame.loc[term, "Std. Err."]),
                    statistic=float(frame.loc[term, "z"]),
                    p_value=float(frame.loc[term, "Pr(>|z|)"]),
                    confidence_interval_lower=float(frame.loc[term, "Conf. Int. Low"]),
                    confidence_interval_upper=float(
                        frame.loc[term, "Cont. Int. Hi."]
                        if "Cont. Int. Hi." in frame.columns
                        else frame.loc[term, "Conf. Int. Hi."]
                    ),
                    magnitude=None,
                    interpretation="Average marginal effect on event probability from a binary probit model.",
                )
            )
    except (AttributeError, KeyError, ValueError, np.linalg.LinAlgError) as error:
        warnings.append(f"Average marginal effects could not be computed: {error}")

    return EffectSizeReport(
        model_id=result.model_id,
        model_type=result.model_type,
        effects=effects,
        model_effects={
            "mcfadden_pseudo_r_squared": result.fit_statistics.get("pseudo_r_squared_mcfadden"),
            "brier_score": result.fit_statistics.get("brier_score"),
        },
        warnings=warnings,
        metadata={"sample_size": result.sample_size, "link": result.metadata.get("link")},
    )


def _build_ordered_logit_effects(
    result: RegressionResult,
) -> EffectSizeReport:
    effects: list[EffectSizeResult] = []

    for coefficient in result.coefficients:
        if "/" in coefficient.term:
            continue

        effects.append(
            EffectSizeResult(
                term=coefficient.term,
                effect_type="odds_ratio",
                estimate=coefficient.exponentiated_estimate,
                standard_error=None,
                statistic=coefficient.statistic,
                p_value=coefficient.p_value,
                confidence_interval_lower=float(np.exp(coefficient.confidence_interval_lower)),
                confidence_interval_upper=float(np.exp(coefficient.confidence_interval_upper)),
                magnitude=None,
                interpretation=("더 높은 종속변수 범주에 속할 누적 오즈의 배수입니다."),
            )
        )

    return EffectSizeReport(
        model_id=result.model_id,
        model_type=result.model_type,
        effects=effects,
        model_effects={},
        warnings=["순서형 로짓의 범주별 한계효과는 후속 버전에서 지원합니다."],
        metadata={
            "sample_size": result.sample_size,
        },
    )


def _coefficient_base_term(term: str) -> str:
    return term.rsplit("::", 1)[-1]


def _is_intercept_term(term: str) -> bool:
    return _coefficient_base_term(term).lower() in {"const", "intercept"}


def _build_count_effects(
    result: RegressionResult,
) -> EffectSizeReport:
    """계수형 회귀의 발생률비 효과크기를 생성한다."""
    effects: list[EffectSizeResult] = []

    for coefficient in result.coefficients:
        if coefficient.term.lower() in {"const", "intercept"} or coefficient.term.startswith(
            "inflate_"
        ):
            continue

        incidence_rate_ratio = coefficient.exponentiated_estimate

        effects.append(
            EffectSizeResult(
                term=coefficient.term,
                effect_type="incidence_rate_ratio",
                estimate=incidence_rate_ratio,
                standard_error=None,
                statistic=coefficient.statistic,
                p_value=coefficient.p_value,
                confidence_interval_lower=float(np.exp(coefficient.confidence_interval_lower)),
                confidence_interval_upper=float(np.exp(coefficient.confidence_interval_upper)),
                magnitude=None,
                interpretation=("독립변수가 1단위 증가할 때 기대 사건 수의 배수입니다."),
            )
        )

    warnings: list[str] = []
    dispersion_ratio = result.fit_statistics.get("dispersion_ratio")
    if (
        dispersion_ratio is not None
        and np.isfinite(float(dispersion_ratio))
        and float(dispersion_ratio) > 1.5
    ):
        warnings.append("과산포 가능성이 있어 발생률비 해석과 표준오차를 주의해서 검토해야 합니다.")

    return EffectSizeReport(
        model_id=result.model_id,
        model_type=result.model_type,
        effects=effects,
        model_effects={
            "deviance_pseudo_r_squared": result.fit_statistics.get("pseudo_r_squared_deviance"),
            "dispersion_ratio": dispersion_ratio,
        },
        warnings=warnings,
        metadata={
            "sample_size": result.sample_size,
        },
    )


def _build_multinomial_logit_effects(result: RegressionResult) -> EffectSizeReport:
    effects: list[EffectSizeResult] = []
    for coefficient in result.coefficients:
        if _is_intercept_term(coefficient.term):
            continue
        effects.append(
            EffectSizeResult(
                term=coefficient.term,
                effect_type="odds_ratio",
                estimate=coefficient.exponentiated_estimate,
                standard_error=None,
                statistic=coefficient.statistic,
                p_value=coefficient.p_value,
                confidence_interval_lower=float(np.exp(coefficient.confidence_interval_lower)),
                confidence_interval_upper=float(np.exp(coefficient.confidence_interval_upper)),
                magnitude=None,
                interpretation="Category-specific odds ratio relative to the multinomial reference category.",
            )
        )

    return EffectSizeReport(
        model_id=result.model_id,
        model_type=result.model_type,
        effects=effects,
        model_effects={
            "mcfadden_pseudo_r_squared": result.fit_statistics.get("pseudo_r_squared_mcfadden"),
            "category_count": result.fit_statistics.get("category_count"),
        },
        metadata={
            "sample_size": result.sample_size,
            "reference_category": result.metadata.get("reference_category"),
            "category_labels": result.metadata.get("category_labels"),
        },
    )


def _build_mixed_effects(
    result: RegressionResult,
) -> EffectSizeReport:
    """Random Intercept 혼합효과 모형의 효과크기를 계산한다."""
    fitted = result.raw_result

    if fitted is None:
        raise ValueError("원본 혼합효과 회귀 결과 객체가 없습니다.")

    endog = np.asarray(fitted.model.endog, dtype=float)
    exog = np.asarray(fitted.model.exog, dtype=float)
    exog_names = [str(name) for name in fitted.model.exog_names]
    fixed_effects = np.asarray(fitted.fe_params, dtype=float)

    outcome_sd = float(np.std(endog, ddof=1))
    if not np.isfinite(outcome_sd) or np.isclose(outcome_sd, 0.0):
        raise ValueError("종속변수 표준편차가 0이므로 혼합효과 표준화 계수를 계산할 수 없습니다.")

    coefficient_lookup = {coefficient.term: coefficient for coefficient in result.coefficients}
    effects: list[EffectSizeResult] = []

    for index, term in enumerate(exog_names):
        if term.lower() in {"const", "intercept"}:
            continue

        coefficient = coefficient_lookup.get(term)
        if coefficient is None:
            continue

        predictor_sd = float(np.std(exog[:, index], ddof=1))
        standardized_beta = coefficient.estimate * predictor_sd / outcome_sd

        effects.append(
            EffectSizeResult(
                term=term,
                effect_type="standardized_beta",
                estimate=float(standardized_beta),
                standard_error=None,
                statistic=coefficient.statistic,
                p_value=coefficient.p_value,
                confidence_interval_lower=None,
                confidence_interval_upper=None,
                magnitude=None,
                interpretation=(
                    "독립변수가 1표준편차 증가할 때 종속변수가 "
                    f"{standardized_beta:.3f}표준편차 변화합니다. "
                    "Random Intercept를 포함한 고정효과 기준 값입니다."
                ),
            )
        )

    fixed_linear_predictor = exog @ fixed_effects
    fixed_effect_variance = float(np.var(fixed_linear_predictor, ddof=1))
    random_intercept_variance = float(fitted.cov_re.iloc[0, 0])
    random_slope_variance = result.fit_statistics.get("random_slope_variance")
    random_intercept_slope_covariance = result.fit_statistics.get(
        "random_intercept_slope_covariance"
    )
    random_effect_variance = random_intercept_variance
    if result.model_type == "mixed_random_slope":
        random_design = np.asarray(fitted.model.exog_re, dtype=float)
        covariance_matrix = np.asarray(fitted.cov_re, dtype=float)
        random_effect_variance = float(
            np.mean(np.einsum("ij,jk,ik->i", random_design, covariance_matrix, random_design))
        )
    residual_variance = float(fitted.scale)
    total_variance = fixed_effect_variance + random_effect_variance + residual_variance

    if not np.isfinite(total_variance) or total_variance <= 0:
        raise ValueError("혼합효과 모형의 총분산을 계산할 수 없습니다.")

    marginal_r_squared = fixed_effect_variance / total_variance
    conditional_r_squared = (fixed_effect_variance + random_effect_variance) / total_variance
    icc_denominator = random_intercept_variance + residual_variance
    intraclass_correlation = (
        random_intercept_variance / icc_denominator if icc_denominator > 0 else None
    )

    warnings: list[str] = []
    if random_intercept_variance <= np.finfo(float).eps:
        warnings.append(
            "Random Intercept 분산이 0에 가까워 conditional R²와 marginal R²의 차이가 작습니다."
        )

    if conditional_r_squared + 1e-12 < marginal_r_squared:
        warnings.append(
            "conditional R²가 marginal R²보다 작게 계산되어 분산 성분을 확인해야 합니다."
        )

    return EffectSizeReport(
        model_id=result.model_id,
        model_type=result.model_type,
        effects=effects,
        model_effects={
            "intraclass_correlation": (
                float(intraclass_correlation) if intraclass_correlation is not None else None
            ),
            "marginal_r_squared": float(marginal_r_squared),
            "conditional_r_squared": float(conditional_r_squared),
            "fixed_effect_variance": fixed_effect_variance,
            "random_intercept_variance": random_intercept_variance,
            "random_slope_variance": random_slope_variance,
            "random_intercept_slope_covariance": random_intercept_slope_covariance,
            "random_intercept_slope_correlation": result.fit_statistics.get(
                "random_intercept_slope_correlation"
            ),
            "random_effect_variance": random_effect_variance,
            "residual_variance": residual_variance,
        },
        warnings=warnings,
        metadata={
            "sample_size": result.sample_size,
            "group_variable": result.metadata.get("group_variable"),
            "group_count": result.fit_statistics.get("group_count"),
            "r_squared_method": "nakagawa_schielzeth",
        },
    )



def _build_gee_effects(result: RegressionResult) -> EffectSizeReport:
    diagnostics = result.metadata.get("diagnostics", {})
    effects: list[EffectSizeResult] = []
    warnings: list[str] = []

    if result.model_type == "gee_gaussian":
        endog = np.asarray(diagnostics.get("endog", []), dtype=float)
        exog = np.asarray(diagnostics.get("exog", []), dtype=float)
        exog_names = [str(name) for name in diagnostics.get("exog_names", [])]
        outcome_sd = float(np.std(endog, ddof=1)) if endog.size > 1 else np.nan
        if not np.isfinite(outcome_sd) or np.isclose(outcome_sd, 0.0):
            warnings.append("GEE standardized effects could not be computed because outcome SD is zero.")
        coefficient_lookup = {coefficient.term: coefficient for coefficient in result.coefficients}
        for index, term in enumerate(exog_names):
            if term.lower() in {"const", "intercept"}:
                continue
            coefficient = coefficient_lookup.get(term)
            if coefficient is None or not np.isfinite(outcome_sd) or np.isclose(outcome_sd, 0.0):
                continue
            predictor_sd = float(np.std(exog[:, index], ddof=1)) if exog.shape[0] > 1 else np.nan
            estimate = coefficient.estimate * predictor_sd / outcome_sd
            effects.append(
                EffectSizeResult(
                    term=term,
                    effect_type="standardized_beta",
                    estimate=float(estimate),
                    standard_error=None,
                    statistic=coefficient.statistic,
                    p_value=coefficient.p_value,
                    confidence_interval_lower=None,
                    confidence_interval_upper=None,
                    magnitude=None,
                    interpretation="Population-averaged standardized GEE coefficient.",
                )
            )
    else:
        effect_type = "odds_ratio" if result.model_type == "gee_logit" else "incidence_rate_ratio"
        interpretation = (
            "Population-averaged odds ratio from GEE."
            if result.model_type == "gee_logit"
            else "Population-averaged incidence rate ratio from GEE."
        )
        for coefficient in result.coefficients:
            if coefficient.term.lower() in {"const", "intercept"}:
                continue
            effects.append(
                EffectSizeResult(
                    term=coefficient.term,
                    effect_type=effect_type,
                    estimate=coefficient.exponentiated_estimate,
                    standard_error=None,
                    statistic=coefficient.statistic,
                    p_value=coefficient.p_value,
                    confidence_interval_lower=float(np.exp(coefficient.confidence_interval_lower)),
                    confidence_interval_upper=float(np.exp(coefficient.confidence_interval_upper)),
                    magnitude=None,
                    interpretation=interpretation,
                )
            )

    return EffectSizeReport(
        model_id=result.model_id,
        model_type=result.model_type,
        effects=effects,
        model_effects={
            "cluster_count": result.fit_statistics.get("cluster_count"),
            "mean_cluster_size": result.fit_statistics.get("mean_cluster_size"),
        },
        warnings=warnings,
        metadata={
            "sample_size": result.sample_size,
            "group_variable": result.metadata.get("group_variable"),
            "covariance_structure": result.metadata.get("covariance_structure"),
        },
    )


def build_regression_effect_size_report(
    result: RegressionResult,
) -> EffectSizeReport:
    """회귀모형 종류에 맞는 효과크기 보고서를 생성한다."""
    if result.model_type == "ols":
        return _build_ols_effects(result)

    if result.model_type == "cox_proportional_hazards":
        return _build_cox_effects(result)

    if result.model_type == "fractional_logit":
        return _build_fractional_logit_effects(result)

    if result.model_type == "beta_regression":
        return _build_beta_regression_effects(result)

    if result.model_type == "gamma_regression":
        return _build_gamma_effects(result)

    if result.model_type == "inverse_gaussian_regression":
        return _build_inverse_gaussian_effects(result)

    if result.model_type == "iv_2sls_regression":
        return _build_iv_2sls_effects(result)

    if result.model_type == "heckman_selection":
        return _build_heckman_effects(result)

    if result.model_type == "quantile_regression":
        return _build_quantile_effects(result)

    if result.model_type == "regularized_regression":
        return _build_regularized_effects(result)

    if result.model_type == "robust_regression":
        return _build_robust_effects(result)

    if result.model_type == "tobit_regression":
        return _build_tobit_effects(result)

    if result.model_type == "panel_fixed_effects":
        return _build_panel_fixed_effects(result)

    if result.model_type == "binary_probit":
        return _build_binary_probit_effects(result)

    if result.model_type in {
        "binary_logit",
        "mixed_binary_logit_random_intercept",
        "mixed_binary_logit_random_slope",
        "mixed_binary_logit_three_level",
    }:
        return _build_binary_logit_effects(result)

    if result.model_type == "ordered_logit":
        return _build_ordered_logit_effects(result)

    if result.model_type == "multinomial_logit":
        return _build_multinomial_logit_effects(result)

    if result.model_type in {"gee_gaussian", "gee_logit", "gee_poisson"}:
        return _build_gee_effects(result)

    if result.model_type in {"mixed_random_intercept", "mixed_random_slope", "mixed_three_level"}:
        return _build_mixed_effects(result)

    if result.model_type in {
        "poisson",
        "negative_binomial",
        "zero_inflated_poisson",
        "zero_inflated_negative_binomial",
        "mixed_poisson_random_intercept",
        "mixed_poisson_random_slope",
        "mixed_poisson_three_level",
        "mixed_negative_binomial_random_intercept",
            "mixed_negative_binomial_random_slope",
            "mixed_negative_binomial_three_level",
    }:
        return _build_count_effects(result)

    raise ValueError(f"지원하지 않는 회귀모형 유형입니다: {result.model_type}")


def effect_size_report_to_dataframe(
    report: EffectSizeReport,
) -> pd.DataFrame:
    """효과크기 결과를 데이터프레임으로 변환한다."""
    return pd.DataFrame([asdict(item) for item in report.effects])


def effect_size_summary_to_dataframe(
    report: EffectSizeReport,
) -> pd.DataFrame:
    """효과크기 보고서 요약을 세로형 표로 변환한다."""
    values = {
        "model_id": report.model_id,
        "model_type": report.model_type,
        "effect_count": len(report.effects),
        "warning_count": len(report.warnings),
        **report.model_effects,
        **report.metadata,
    }

    return pd.DataFrame(
        {
            "item": list(values.keys()),
            "value": list(values.values()),
        }
    )
