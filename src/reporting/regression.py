"""회귀결과를 논문용 표와 한국어 결과문으로 변환한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.statistics.regression.base import RegressionResult

_MIXED_BINARY_LOGIT_MODELS = {
    "mixed_binary_logit_random_intercept",
    "mixed_binary_logit_random_slope",
    "mixed_binary_logit_three_level",
}

_MIXED_POISSON_MODELS = {
    "mixed_poisson_random_intercept",
    "mixed_poisson_random_slope",
    "mixed_poisson_three_level",
}

_MIXED_NEGATIVE_BINOMIAL_MODELS = {
    "mixed_negative_binomial_random_intercept",
    "mixed_negative_binomial_random_slope",
    "mixed_negative_binomial_three_level",
}

_THREE_LEVEL_GLMM_MODELS = {
    "mixed_binary_logit_three_level",
    "mixed_poisson_three_level",
    "mixed_negative_binomial_three_level",
}

_RANDOM_SLOPE_GLMM_MODELS = {
    "mixed_binary_logit_random_slope",
    "mixed_poisson_random_slope",
    "mixed_negative_binomial_random_slope",
}

_GLMM_MODELS = (
    _MIXED_BINARY_LOGIT_MODELS
    | _MIXED_POISSON_MODELS
    | _MIXED_NEGATIVE_BINOMIAL_MODELS
)


@dataclass(slots=True)
class RegressionPublicationReport:
    """논문용 회귀결과 보고서."""

    model_id: str
    model_type: str
    publication_table: pd.DataFrame
    model_summary: pd.DataFrame
    narrative: str
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _significance_stars(p_value: float | None) -> str:
    if p_value is None:
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _coefficient_base_term(term: str) -> str:
    return term.rsplit("::", 1)[-1]


def _is_intercept_term(term: str) -> bool:
    return _coefficient_base_term(term).lower() in {"const", "intercept"}


def _effect_lookup(effect_report: Any | None) -> dict[tuple[str, str], float]:
    lookup: dict[tuple[str, str], float] = {}

    if effect_report is None:
        return lookup

    for effect in effect_report.effects:
        if effect.estimate is not None:
            lookup[(effect.term, effect.effect_type)] = float(effect.estimate)

    return lookup


def build_publication_table(
    regression_result: RegressionResult,
    effect_report: Any | None = None,
) -> pd.DataFrame:
    """회귀계수와 효과크기를 하나의 논문용 표로 결합한다."""
    effect_lookup = _effect_lookup(effect_report)
    rows: list[dict[str, Any]] = []

    for coefficient in regression_result.coefficients:
        if "/" in coefficient.term:
            term_type = "threshold"
        elif regression_result.model_type == "beta_regression" and coefficient.term == "precision":
            term_type = "ancillary"
        elif coefficient.term.startswith("inflate_"):
            term_type = "inflation"
        elif _is_intercept_term(coefficient.term):
            term_type = "intercept"
        else:
            term_type = "predictor"

        row = {
            "변수": coefficient.term,
            "구분": term_type,
            "계수": coefficient.estimate,
            "표준오차": coefficient.standard_error,
            "검정통계량": coefficient.statistic,
            "p값": coefficient.p_value,
            "95% CI 하한": coefficient.confidence_interval_lower,
            "95% CI 상한": coefficient.confidence_interval_upper,
            "유의성": _significance_stars(coefficient.p_value),
            "표준화 β": effect_lookup.get(
                (
                    coefficient.term,
                    "standardized_beta",
                )
            ),
            "부분 R²": effect_lookup.get(
                (
                    coefficient.term,
                    "partial_r_squared",
                )
            ),
            "부분 Cohen f²": effect_lookup.get(
                (
                    coefficient.term,
                    "partial_cohen_f_squared",
                )
            ),
            "오즈비": effect_lookup.get(
                (
                    coefficient.term,
                    "odds_ratio",
                )
            ),
            "평균한계효과": effect_lookup.get(
                (
                    coefficient.term,
                    "average_marginal_effect",
                )
            ),
            "발생률비": effect_lookup.get(
                (
                    coefficient.term,
                    "incidence_rate_ratio",
                )
            ),
            "hazard_ratio": effect_lookup.get(
                (
                    coefficient.term,
                    "hazard_ratio",
                )
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_model_summary(
    regression_result: RegressionResult,
    effect_report: Any | None = None,
) -> pd.DataFrame:
    """모형 적합도와 전체 효과크기를 세로형 표로 생성한다."""
    values: dict[str, Any] = {
        "모형 ID": regression_result.model_id,
        "모형 유형": regression_result.model_type,
        "종속변수": regression_result.dependent_variable,
        "표본 수": regression_result.sample_size,
        "수렴 여부": regression_result.converged,
        "표준오차 유형": regression_result.standard_error_type,
    }

    values.update(regression_result.fit_statistics)

    if effect_report is not None:
        values.update(effect_report.model_effects)

    return pd.DataFrame(
        {
            "항목": list(values.keys()),
            "값": list(values.values()),
        }
    )


def _direction_text(estimate: float) -> str:
    if estimate > 0:
        return "정(+)의"
    if estimate < 0:
        return "부(-)의"
    return "0에 가까운"


def _format_p_value(p_value: float) -> str:
    if p_value < 0.001:
        return "p<.001"
    return f"p={p_value:.3f}"




def _append_glmm_structure_sentences(
    sentences: list[str],
    regression_result: RegressionResult,
) -> None:
    if regression_result.model_type in _THREE_LEVEL_GLMM_MODELS:
        level2_group = regression_result.metadata.get("level2_group", "Level 2")
        level3_group = regression_result.metadata.get("level3_group", "Level 3")
        level2_count = regression_result.fit_statistics.get("level2_group_count")
        level3_count = regression_result.fit_statistics.get("level3_group_count")
        if level2_count is not None and level3_count is not None:
            sentences.append(
                f"3-level GLMM included {int(level2_count)} {level2_group} groups "
                f"nested within {int(level3_count)} {level3_group} groups."
            )
        level2_vpc = regression_result.fit_statistics.get("level2_vpc")
        level3_vpc = regression_result.fit_statistics.get("level3_vpc")
        if level2_vpc is not None and level3_vpc is not None:
            sentences.append(
                f"Variance partition coefficients were Level 2={float(level2_vpc):.3f} "
                f"and Level 3={float(level3_vpc):.3f}."
            )
    else:
        group_variable = regression_result.metadata.get("group_variable")
        group_count = regression_result.fit_statistics.get("group_count")
        if group_variable is not None and group_count is not None:
            sentences.append(
                f"The GLMM included {int(group_count)} groups defined by {group_variable}."
            )

    if regression_result.model_type in _RANDOM_SLOPE_GLMM_MODELS:
        slope = regression_result.metadata.get("random_slope_variable")
        slope_variance = regression_result.fit_statistics.get("random_slope_variance")
        if slope is not None and slope_variance is not None:
            sentences.append(
                f"A random slope for {slope} was estimated "
                f"(variance={float(slope_variance):.3f})."
            )

    intercept_variance = regression_result.fit_statistics.get("random_intercept_variance")
    if intercept_variance is not None:
        sentences.append(f"Random intercept variance was {float(intercept_variance):.3f}.")

    icc = regression_result.fit_statistics.get("icc")
    if icc is not None:
        sentences.append(f"The latent-scale ICC was {float(icc):.3f}.")

    alpha = regression_result.fit_statistics.get("dispersion_alpha")
    if alpha is not None:
        sentences.append(f"The NB2 dispersion parameter alpha was {float(alpha):.3f}.")

    sentences.append(
        "The GLMM converged."
        if regression_result.converged
        else "The GLMM did not converge; interpretation requires caution."
    )

def write_korean_results_narrative(
    regression_result: RegressionResult,
    effect_report: Any | None = None,
) -> str:
    """논문 결과 절에 사용할 한국어 서술 초안을 생성한다."""
    effect_lookup = _effect_lookup(effect_report)
    substantive = [
        coefficient
        for coefficient in regression_result.coefficients
        if not _is_intercept_term(coefficient.term)
        and "/" not in coefficient.term
        and not coefficient.term.startswith("inflate_")
        and not (regression_result.model_type == "beta_regression" and coefficient.term == "precision")
    ]

    sentences: list[str] = []

    cross_level = regression_result.metadata.get("cross_level_interaction")

    model_name = {
        "ols": "OLS 회귀분석",
        "binary_logit": "이항 로지스틱 회귀분석",
        "binary_cloglog": "Binary complementary log-log regression",
        "binary_probit": "Binary probit regression",
        "multinomial_logit": "Multinomial logistic regression",
        "ordered_logit": "순서형 로지스틱 회귀분석",
        "ordered_probit": "Ordered probit regression",
        "poisson": "포아송 회귀분석",
        "negative_binomial": "음이항 회귀분석",
        "zero_inflated_poisson": "영과잉 포아송 회귀분석",
        "zero_inflated_negative_binomial": "영과잉 음이항 회귀분석",
        "mixed_random_intercept": "Random Intercept 혼합효과모형",
        "mixed_binary_logit_random_intercept": "Random Intercept 혼합 이항 로지스틱 회귀분석",
        "mixed_binary_logit_random_slope": "Random Slope 혼합 이항 로지스틱 회귀분석",
        "mixed_binary_logit_three_level": "3수준 혼합 이항 로지스틱 회귀분석",
        "mixed_poisson_random_intercept": "Random Intercept 혼합 포아송 회귀분석",
        "mixed_poisson_random_slope": "Random Slope 혼합 포아송 회귀분석",
        "mixed_poisson_three_level": "3수준 혼합 포아송 회귀분석",
        "mixed_negative_binomial_random_intercept": "Random Intercept 혼합 음이항 회귀분석",
        "mixed_negative_binomial_random_slope": "Random Slope 혼합 음이항 회귀분석",
        "mixed_negative_binomial_three_level": "3수준 혼합 음이항 회귀분석",
        "mixed_random_slope": "Random Slope 혼합효과모형",
        "mixed_three_level": "3수준 혼합효과모형",
    }.get(
        regression_result.model_type,
        regression_result.model_type,
    )

    sentences.append(
        f"{model_name}을 실시한 결과, 분석에 사용된 표본은 {regression_result.sample_size}개였다."
    )

    for coefficient in substantive:
        significant = coefficient.p_value < 0.05
        direction = _direction_text(coefficient.estimate)
        p_text = _format_p_value(coefficient.p_value)

        if regression_result.model_type in {"ols", "heckman_selection", "iv_2sls_regression", "regularized_regression", "robust_regression", "quantile_regression", "tobit_regression", "panel_fixed_effects", "mixed_random_intercept", "mixed_random_slope", "gee_gaussian"}:
            beta = (
                effect_lookup.get(
                    (
                        coefficient.term,
                        "standardized_beta",
                    )
                )
                or effect_lookup.get(
                    (
                        coefficient.term,
                        "standardized_quantile_beta",
                    )
                )
                or effect_lookup.get(
                    (
                        coefficient.term,
                        "within_standardized_beta",
                    )
                )
                or effect_lookup.get(
                    (
                        coefficient.term,
                        "latent_standardized_beta",
                    )
                )
                or effect_lookup.get(
                    (
                        coefficient.term,
                        "robust_standardized_beta",
                    )
                )
                or effect_lookup.get(
                    (
                        coefficient.term,
                        "regularized_standardized_beta",
                    )
                )
                or effect_lookup.get(
                    (
                        coefficient.term,
                        "heckman_standardized_beta",
                    )
                )
                or effect_lookup.get(
                    (
                        coefficient.term,
                        "iv_standardized_beta",
                    )
                )
            )

            if beta is not None:
                effect_text = f"β={beta:.3f}"
            else:
                effect_text = f"B={coefficient.estimate:.3f}"

        elif regression_result.model_type == "beta_regression":
            mean_or = effect_lookup.get(
                (
                    coefficient.term,
                    "mean_odds_ratio",
                )
            )
            effect_text = (
                f"MOR={mean_or:.3f}"
                if mean_or is not None
                else f"B={coefficient.estimate:.3f}"
            )
        elif regression_result.model_type in {"gamma_regression", "inverse_gaussian_regression"}:
            mean_ratio = effect_lookup.get(
                (
                    coefficient.term,
                    "mean_ratio",
                )
            )
            effect_text = (
                f"MR={mean_ratio:.3f}"
                if mean_ratio is not None
                else f"B={coefficient.estimate:.3f}"
            )
        elif regression_result.model_type == "fractional_logit":
            fractional_or = effect_lookup.get(
                (
                    coefficient.term,
                    "fractional_odds_ratio",
                )
            )
            effect_text = (
                f"FOR={fractional_or:.3f}"
                if fractional_or is not None
                else f"B={coefficient.estimate:.3f}"
            )
        elif regression_result.model_type == "cox_proportional_hazards":
            hazard_ratio = effect_lookup.get(
                (
                    coefficient.term,
                    "hazard_ratio",
                )
            )
            effect_text = (
                f"HR={hazard_ratio:.3f}"
                if hazard_ratio is not None
                else f"B={coefficient.estimate:.3f}"
            )
        elif regression_result.model_type == "binary_cloglog":
            hazard_ratio = effect_lookup.get((coefficient.term, "cloglog_hazard_ratio"))
            effect_text = (
                f"HR={hazard_ratio:.3f}"
                if hazard_ratio is not None
                else f"B={coefficient.estimate:.3f}"
            )
        elif regression_result.model_type == "binary_probit":
            marginal_effect = effect_lookup.get((coefficient.term, "average_marginal_effect"))
            effect_text = (
                f"AME={marginal_effect:.3f}"
                if marginal_effect is not None
                else f"B={coefficient.estimate:.3f}"
            )
        elif regression_result.model_type == "ordered_probit":
            latent = effect_lookup.get((coefficient.term, "ordered_probit_latent_coefficient"))
            effect_text = (
                f"latent B={latent:.3f}"
                if latent is not None
                else f"B={coefficient.estimate:.3f}"
            )
        elif regression_result.model_type in {
            "binary_logit",
            "mixed_binary_logit_random_intercept",
            "mixed_binary_logit_random_slope",
            "mixed_binary_logit_three_level",
            "ordered_logit",
            "multinomial_logit",
            "gee_logit",
        }:
            odds_ratio = effect_lookup.get(
                (
                    coefficient.term,
                    "odds_ratio",
                )
            )
            effect_text = (
                f"OR={odds_ratio:.3f}"
                if odds_ratio is not None
                else f"B={coefficient.estimate:.3f}"
            )
        elif regression_result.model_type in {
            "poisson",
            "negative_binomial",
            "mixed_poisson_random_intercept",
            "mixed_poisson_random_slope",
            "mixed_poisson_three_level",
            "mixed_negative_binomial_random_intercept",
            "mixed_negative_binomial_random_slope",
            "mixed_negative_binomial_three_level",
            "gee_poisson",
        }:
            incidence_rate_ratio = effect_lookup.get(
                (
                    coefficient.term,
                    "incidence_rate_ratio",
                )
            )
            effect_text = (
                f"IRR={incidence_rate_ratio:.3f}"
                if incidence_rate_ratio is not None
                else f"B={coefficient.estimate:.3f}"
            )
        else:
            effect_text = f"B={coefficient.estimate:.3f}"

        if significant:
            sentences.append(
                f"{coefficient.term}은 종속변수에 유의한 "
                f"{direction} 영향을 보였다"
                f"({effect_text}, {p_text})."
            )
        else:
            sentences.append(
                f"{coefficient.term}의 효과는 통계적으로 유의하지 않았다({effect_text}, {p_text})."
            )

    if regression_result.model_type == "ols":
        r_squared = regression_result.fit_statistics.get("r_squared")
        adjusted = regression_result.fit_statistics.get("adjusted_r_squared")

        if r_squared is not None:
            if adjusted is not None:
                sentences.append(
                    f"모형의 설명력은 R²={r_squared:.3f}, 수정 R²={adjusted:.3f}이었다."
                )
            else:
                sentences.append(f"모형의 설명력은 R²={r_squared:.3f}이었다.")

    elif regression_result.model_type == "quantile_regression":
        quantile = regression_result.fit_statistics.get("quantile")
        pseudo = regression_result.fit_statistics.get("pseudo_r_squared")
        pinball = regression_result.fit_statistics.get("pinball_loss")
        if quantile is not None:
            sentences.append(f"The modeled conditional quantile was q={float(quantile):.2f}.")
        if pseudo is not None:
            sentences.append(f"Quantile pseudo R-squared was {float(pseudo):.3f}.")
        if pinball is not None:
            sentences.append(f"Mean pinball loss was {float(pinball):.3f}.")

    elif regression_result.model_type == "heckman_selection":
        selection_variable = regression_result.metadata.get("selection_variable")
        selection_rate = regression_result.fit_statistics.get("selection_rate")
        inverse_mills = regression_result.fit_statistics.get("inverse_mills_coefficient")
        inverse_mills_p = regression_result.fit_statistics.get("inverse_mills_p_value")
        exclusions = regression_result.metadata.get("exclusion_restrictions") or []
        if selection_variable is not None:
            sentences.append(f"Heckman selection modeled observation through {selection_variable}.")
        if selection_rate is not None:
            sentences.append(f"The observed outcome selection rate was {float(selection_rate):.3f}.")
        if exclusions:
            sentences.append(
                "Selection exclusion restrictions were "
                + ", ".join(str(value) for value in exclusions)
                + "."
            )
        if inverse_mills is not None and inverse_mills_p is not None:
            sentences.append(
                f"The inverse Mills ratio was {float(inverse_mills):.3f} "
                f"({_format_p_value(float(inverse_mills_p))})."
            )

    elif regression_result.model_type == "iv_2sls_regression":
        endogenous = regression_result.metadata.get("endogenous_variables") or []
        instruments = regression_result.metadata.get("instrument_variables") or []
        min_f = regression_result.fit_statistics.get("minimum_first_stage_f_statistic")
        r_squared = regression_result.fit_statistics.get("r_squared")
        sentences.append(
            "IV 2SLS treated "
            + ", ".join(str(value) for value in endogenous)
            + " as endogenous and used instruments "
            + ", ".join(str(value) for value in instruments)
            + "."
        )
        if min_f is not None:
            sentences.append(f"Minimum first-stage excluded-instrument F statistic was {float(min_f):.3f}.")
        if r_squared is not None:
            sentences.append(f"Second-stage R-squared was {float(r_squared):.3f}.")

    elif regression_result.model_type == "regularized_regression":
        penalty = regression_result.fit_statistics.get("penalty")
        alpha = regression_result.fit_statistics.get("alpha")
        l1_ratio = regression_result.fit_statistics.get("l1_ratio")
        selected = regression_result.fit_statistics.get("selected_coefficient_count")
        zero = regression_result.fit_statistics.get("zero_coefficient_count")
        rmse = regression_result.fit_statistics.get("root_mean_squared_error")
        if penalty is not None:
            sentences.append(
                f"Regularized regression used {penalty} penalty "
                f"(alpha={float(alpha):.3f}, l1_ratio={float(l1_ratio):.2f})."
            )
        if selected is not None and zero is not None:
            sentences.append(
                f"The penalty retained {int(selected)} coefficients and shrank {int(zero)} to zero."
            )
        if rmse is not None:
            sentences.append(f"Regularized prediction RMSE was {float(rmse):.3f}.")

    elif regression_result.model_type == "robust_regression":
        pseudo = regression_result.fit_statistics.get("pseudo_r_squared")
        downweighted = regression_result.fit_statistics.get("downweighted_count")
        heavy = regression_result.fit_statistics.get("heavily_downweighted_count")
        norm = regression_result.metadata.get("norm")
        if norm is not None:
            sentences.append(f"Robust regression used {norm} M-estimation weights.")
        if pseudo is not None:
            sentences.append(f"Robust observed-scale pseudo R-squared was {float(pseudo):.3f}.")
        if downweighted is not None and heavy is not None:
            sentences.append(
                f"The robust fit downweighted {int(downweighted)} observations; "
                f"{int(heavy)} received weights below 0.5."
            )

    elif regression_result.model_type == "tobit_regression":
        lower_limit = regression_result.metadata.get("lower_limit")
        upper_limit = regression_result.metadata.get("upper_limit")
        left_count = regression_result.fit_statistics.get("left_censored_count")
        right_count = regression_result.fit_statistics.get("right_censored_count")
        censoring_rate = regression_result.fit_statistics.get("censoring_rate")
        pseudo = regression_result.fit_statistics.get("pseudo_r_squared")
        sigma = regression_result.fit_statistics.get("sigma")
        if lower_limit is not None or upper_limit is not None:
            sentences.append(
                f"Tobit censoring limits were lower={lower_limit} and upper={upper_limit}."
            )
        if left_count is not None and right_count is not None and censoring_rate is not None:
            sentences.append(
                f"Censored observations included {int(left_count)} left-censored and "
                f"{int(right_count)} right-censored cases "
                f"({float(censoring_rate):.1%} of the analytic sample)."
            )
        if pseudo is not None:
            sentences.append(f"Observed-scale pseudo R-squared was {float(pseudo):.3f}.")
        if sigma is not None:
            sentences.append(f"Estimated latent residual sigma was {float(sigma):.3f}.")

    elif regression_result.model_type == "panel_fixed_effects":
        entity_variable = regression_result.metadata.get("entity_variable")
        time_variable = regression_result.metadata.get("time_variable")
        entity_count = regression_result.fit_statistics.get("entity_count")
        time_count = regression_result.fit_statistics.get("time_period_count")
        within_r_squared = regression_result.fit_statistics.get("within_r_squared")
        absorbed = regression_result.metadata.get("absorbed_effects") or []
        if entity_count is not None and entity_variable is not None:
            sentences.append(
                f"Panel fixed effects absorbed {int(entity_count)} entities defined by {entity_variable}."
            )
        if time_count is not None and time_variable is not None:
            sentences.append(
                f"Time fixed effects covered {int(time_count)} periods defined by {time_variable}."
            )
        if within_r_squared is not None:
            sentences.append(f"Within R-squared was {float(within_r_squared):.3f}.")
        if absorbed:
            sentences.append("Absorbed fixed effects were " + ", ".join(str(item) for item in absorbed) + ".")

    elif regression_result.model_type == "beta_regression":
        pseudo = regression_result.fit_statistics.get("pseudo_r_squared")
        precision = regression_result.fit_statistics.get("precision")
        rmse = regression_result.fit_statistics.get("root_mean_squared_error")
        if pseudo is not None:
            sentences.append(f"Beta regression pseudo R-squared was {float(pseudo):.3f}.")
        if precision is not None:
            sentences.append(f"Estimated precision was {float(precision):.3f}.")
        if rmse is not None:
            sentences.append(f"Prediction RMSE was {float(rmse):.3f}.")

    elif regression_result.model_type == "inverse_gaussian_regression":
        pseudo = regression_result.fit_statistics.get("pseudo_r_squared_deviance")
        dispersion = regression_result.fit_statistics.get("dispersion_ratio")
        rmse = regression_result.fit_statistics.get("root_mean_squared_error")
        if pseudo is not None:
            sentences.append(f"Inverse Gaussian deviance pseudo R-squared was {float(pseudo):.3f}.")
        if dispersion is not None:
            sentences.append(f"Inverse Gaussian Pearson dispersion ratio was {float(dispersion):.3f}.")
        if rmse is not None:
            sentences.append(f"Inverse Gaussian prediction RMSE was {float(rmse):.3f}.")

    elif regression_result.model_type == "gamma_regression":
        pseudo = regression_result.fit_statistics.get("pseudo_r_squared_deviance")
        dispersion = regression_result.fit_statistics.get("dispersion_ratio")
        rmse = regression_result.fit_statistics.get("root_mean_squared_error")
        if pseudo is not None:
            sentences.append(f"Gamma deviance pseudo R-squared was {float(pseudo):.3f}.")
        if dispersion is not None:
            sentences.append(f"Gamma Pearson dispersion ratio was {float(dispersion):.3f}.")
        if rmse is not None:
            sentences.append(f"Gamma prediction RMSE was {float(rmse):.3f}.")

    elif regression_result.model_type == "fractional_logit":
        pseudo = regression_result.fit_statistics.get("pseudo_r_squared_deviance")
        dispersion = regression_result.fit_statistics.get("dispersion_ratio")
        boundary_count = regression_result.fit_statistics.get("boundary_count")
        if pseudo is not None:
            sentences.append(f"Deviance pseudo R-squared was {float(pseudo):.3f}.")
        if dispersion is not None:
            sentences.append(f"Pearson dispersion ratio was {float(dispersion):.3f}.")
        if boundary_count is not None:
            sentences.append(f"Boundary observations at 0 or 1 numbered {int(boundary_count)}.")

    elif regression_result.model_type == "cox_proportional_hazards":
        event_count = regression_result.fit_statistics.get("event_count")
        censored_count = regression_result.fit_statistics.get("censored_count")
        events_per_parameter = regression_result.fit_statistics.get("events_per_parameter")
        event_variable = regression_result.metadata.get("event_variable")
        if event_count is not None and censored_count is not None:
            sentences.append(
                f"The Cox model included {int(event_count)} events and {int(censored_count)} censored observations."
            )
        if event_variable is not None:
            sentences.append(f"Event status was defined by {event_variable}.")
        if events_per_parameter is not None:
            sentences.append(f"Events per parameter was {float(events_per_parameter):.2f}.")

    elif regression_result.model_type in {
        "mixed_random_intercept",
        "mixed_random_slope",
        "mixed_three_level",
    }:
        marginal = (
            effect_report.model_effects.get("marginal_r_squared")
            if effect_report is not None
            else None
        )
        conditional = (
            effect_report.model_effects.get("conditional_r_squared")
            if effect_report is not None
            else None
        )
        icc = (
            effect_report.model_effects.get("intraclass_correlation")
            if effect_report is not None
            else regression_result.fit_statistics.get("intraclass_correlation")
        )
        group_count = regression_result.fit_statistics.get("group_count")
        covariance_structure = regression_result.metadata.get(
            "random_effect_covariance", "correlated"
        )

        if regression_result.model_type == "mixed_random_slope":
            structure_text = "비상관(대각)" if covariance_structure == "diagonal" else "상관"
            sentences.append(
                f"Random Intercept와 Random Slope에는 {structure_text} 공분산 구조를 적용하였다."
            )

        if regression_result.model_type == "mixed_three_level":
            level2_group = regression_result.metadata.get("level2_group", "Level 2")
            level3_group = regression_result.metadata.get("level3_group", "Level 3")
            level2_count = regression_result.fit_statistics.get("level2_group_count")
            level3_count = regression_result.fit_statistics.get("level3_group_count")
            if level2_count is not None and level3_count is not None:
                sentences.append(
                    f"분석에는 {int(level3_count)}개 {level3_group} 집단과 "
                    f"그 안에 중첩된 {int(level2_count)}개 {level2_group} 집단이 포함되었다."
                )
            level2_icc = regression_result.fit_statistics.get("level2_intraclass_correlation")
            level3_icc = regression_result.fit_statistics.get("level3_intraclass_correlation")
            if level2_icc is not None and level3_icc is not None:
                sentences.append(
                    f"전체 분산 중 {level3_group} 수준 비중은 {float(level3_icc):.3f}, "
                    f"{level2_group} 수준 비중은 {float(level2_icc):.3f}이었다."
                )
        elif group_count is not None:
            sentences.append(f"분석에는 {int(group_count)}개 집단이 포함되었다.")

        if icc is not None:
            sentences.append(f"집단 내 상관계수는 ICC={float(icc):.3f}로 나타났다.")

        if marginal is not None and conditional is not None:
            sentences.append(
                "고정효과의 설명력은 "
                f"marginal R²={float(marginal):.3f}, "
                "고정효과와 Random Intercept를 함께 고려한 설명력은 "
                f"conditional R²={float(conditional):.3f}이었다."
            )

        random_variance = regression_result.fit_statistics.get("random_intercept_variance")
        residual_variance = regression_result.fit_statistics.get("residual_variance")
        if random_variance is not None and residual_variance is not None:
            sentences.append(
                "Random Intercept 분산은 "
                f"{float(random_variance):.3f}, 잔차분산은 "
                f"{float(residual_variance):.3f}이었다."
            )

        if cross_level:
            interaction_term = cross_level.get("interaction_term")
            interaction = next(
                (c for c in regression_result.coefficients if c.term == interaction_term), None
            )
            if interaction is not None:
                sentences.append(
                    f"{cross_level.get('predictor')}와 {cross_level.get('moderator')}의 교차수준 상호작용은 "
                    + ("유의하였다" if interaction.p_value < 0.05 else "유의하지 않았다")
                    + f"(B={interaction.estimate:.3f}, {_format_p_value(interaction.p_value)})."
                )
            slopes = cross_level.get("conditional_effects") or []
            for slope in slopes:
                sentences.append(
                    f"조절변수 {slope['label']} 수준에서 {cross_level.get('predictor')}의 조건부 효과는 "
                    f"B={float(slope['estimate']):.3f}({_format_p_value(float(slope['p_value']))})였다."
                )
            jn = cross_level.get("johnson_neyman")
            if jn is not None:
                roots = jn.get("roots_within_observed_range", [])
                if roots:
                    sentences.append(
                        "Johnson–Neyman 분석에서 관측범위 내 임계값은 "
                        + ", ".join(f"{float(v):.3f}" for v in roots)
                        + "이었다."
                    )
                else:
                    sentences.append(
                        "Johnson–Neyman 분석에서 관측범위 내 유의성 전환 임계값은 확인되지 않았다."
                    )

        sentences.append(
            "모형은 수렴하였다."
            if regression_result.converged
            else "모형이 수렴하지 않아 결과 해석에 주의가 필요하다."
        )

    elif regression_result.model_type == "ordered_probit":
        category_count = regression_result.fit_statistics.get("category_count")
        if category_count is not None:
            sentences.append(f"The ordered probit modeled {int(category_count)} ordinal outcome categories.")

    elif regression_result.model_type in {
        "binary_logit",
        "binary_cloglog",
        "binary_probit",
        "ordered_probit",
        "mixed_binary_logit_random_intercept",
        "mixed_binary_logit_random_slope",
    }:
        pseudo = regression_result.fit_statistics.get("pseudo_r_squared_mcfadden")
        lr_p = regression_result.fit_statistics.get("likelihood_ratio_p_value")

        if pseudo is not None:
            sentences.append(f"McFadden 의사 R²는 {pseudo:.3f}이었다.")

        if lr_p is not None:
            sentences.append(f"우도비 검정의 유의확률은 {_format_p_value(float(lr_p))}였다.")

    elif regression_result.model_type in _GLMM_MODELS:
        _append_glmm_structure_sentences(sentences, regression_result)

    elif regression_result.model_type in {"gee_gaussian", "gee_logit", "gee_poisson"}:
        cluster_count = regression_result.fit_statistics.get("cluster_count")
        group_variable = regression_result.metadata.get("group_variable")
        covariance_structure = regression_result.metadata.get("covariance_structure")
        if cluster_count is not None and group_variable is not None:
            sentences.append(
                f"GEE accounted for {int(cluster_count)} clusters defined by {group_variable}."
            )
        if covariance_structure is not None:
            sentences.append(f"The working correlation structure was {covariance_structure}.")

    elif regression_result.model_type == "multinomial_logit":
        reference_category = regression_result.metadata.get("reference_category")
        category_count = regression_result.fit_statistics.get("category_count")
        pseudo = regression_result.fit_statistics.get("pseudo_r_squared_mcfadden")
        if category_count is not None and reference_category is not None:
            sentences.append(
                f"The nominal outcome had {int(category_count)} categories; {reference_category} was the reference category."
            )
        if pseudo is not None:
            sentences.append(f"McFadden pseudo R-squared was {float(pseudo):.3f}.")

    elif regression_result.model_type == "poisson":
        dispersion_ratio = regression_result.fit_statistics.get("dispersion_ratio")
        pseudo = regression_result.fit_statistics.get("pseudo_r_squared_deviance")

        if pseudo is not None:
            sentences.append(f"Deviance 기반 의사 R²는 {float(pseudo):.3f}이었다.")

        if dispersion_ratio is not None:
            sentences.append(f"Pearson 분산비는 {float(dispersion_ratio):.3f}이었다.")

    elif regression_result.model_type in {
        "negative_binomial",
        "zero_inflated_negative_binomial",
    }:
        alpha = regression_result.fit_statistics.get("alpha")
        pseudo = regression_result.fit_statistics.get("pseudo_r_squared_mcfadden")
        if pseudo is not None:
            sentences.append(f"McFadden 의사 R²는 {float(pseudo):.3f}이었다.")
        if alpha is not None:
            sentences.append(f"음이항 과산포 모수 alpha는 {float(alpha):.3f}이었다.")

    return " ".join(sentences)


def build_regression_publication_report(
    regression_result: RegressionResult,
    effect_report: Any | None = None,
) -> RegressionPublicationReport:
    """논문용 회귀표·요약표·결과문을 한 번에 생성한다."""
    publication_table = build_publication_table(
        regression_result,
        effect_report,
    )
    model_summary = build_model_summary(
        regression_result,
        effect_report,
    )
    narrative = write_korean_results_narrative(
        regression_result,
        effect_report,
    )

    notes = [
        "* p<.05, ** p<.01, *** p<.001.",
    ]

    if regression_result.model_type in {"ols", "heckman_selection", "iv_2sls_regression", "regularized_regression", "robust_regression", "quantile_regression", "tobit_regression", "panel_fixed_effects"}:
        notes.append("OLS의 표준화 β와 부분 효과크기를 함께 제시한다.")
    elif regression_result.model_type in {
        "binary_logit",
        "binary_cloglog",
        "binary_probit",
        "mixed_binary_logit_random_intercept",
        "mixed_binary_logit_random_slope",
        "mixed_binary_logit_three_level",
        "ordered_logit",
        "multinomial_logit",
    }:
        notes.append("로짓 모형은 오즈비를 함께 제시한다.")
    elif regression_result.model_type in {
        "poisson",
        "negative_binomial",
        "mixed_poisson_random_intercept",
        "mixed_poisson_random_slope",
        "mixed_poisson_three_level",
        "mixed_negative_binomial_random_intercept",
            "mixed_negative_binomial_random_slope",
            "mixed_negative_binomial_three_level",
    }:
        notes.append("계수형 회귀모형은 발생률비(IRR)를 함께 제시한다.")
    elif regression_result.model_type in {
        "mixed_random_intercept",
        "mixed_random_slope",
        "mixed_three_level",
    }:
        notes.extend(
            [
                "혼합효과모형의 고정효과는 표준화 β와 함께 제시한다.",
                "marginal R²는 고정효과, conditional R²는 고정효과와 Random Intercept의 설명력을 나타낸다.",
                "ICC는 전체 잔차 변동 중 집단 간 차이가 차지하는 비율이다.",
            ]
        )

    if regression_result.model_type == "beta_regression":
        notes.append("Beta regression models require outcomes strictly inside (0, 1) and report mean odds ratios.")

    if regression_result.model_type == "fractional_logit":
        notes.append("Fractional logit models report fractional odds ratios and average marginal effects when available.")

    if regression_result.model_type == "gamma_regression":
        notes.append("Gamma regression uses a log link and reports multiplicative mean ratios.")

    if regression_result.model_type == "inverse_gaussian_regression":
        notes.append("Inverse Gaussian regression uses a log link and reports multiplicative mean ratios.")

    if regression_result.model_type == "cox_proportional_hazards":
        notes.append("Cox models report hazard ratios from partial likelihood estimates.")

    if regression_result.model_type == "binary_cloglog":
        notes.append("Binary complementary log-log models report exponentiated coefficients and average marginal effects.")

    if regression_result.model_type == "binary_probit":
        notes.append("Binary probit reports latent-index coefficients and average marginal effects when available.")

    if regression_result.model_type == "ordered_probit":
        notes.append("Ordered probit reports latent-index coefficients and ordered threshold parameters.")

    if regression_result.model_type in {"gee_gaussian", "gee_logit", "gee_poisson"}:
        notes.append("GEE models are population-averaged and use robust sandwich standard errors.")

    if regression_result.model_type == "panel_fixed_effects":
        notes.append("Panel fixed-effects models report within-panel coefficients after absorbing entity and optional time effects.")

    if regression_result.model_type == "tobit_regression":
        notes.append("Tobit models estimate latent-scale coefficients for censored continuous outcomes.")

    if regression_result.model_type == "robust_regression":
        notes.append("Robust regression uses M-estimation weights to reduce sensitivity to outlying residuals.")

    if regression_result.model_type == "regularized_regression":
        notes.append("Regularized regression reports penalized coefficients; standard errors and p-values are not inferential.")

    if regression_result.model_type == "iv_2sls_regression":
        notes.append("IV 2SLS reports second-stage coefficients and first-stage instrument strength diagnostics.")

    if regression_result.model_type == "heckman_selection":
        notes.append("Heckman selection reports outcome-equation coefficients with inverse Mills correction and a first-stage selection equation.")

    if regression_result.model_type in _GLMM_MODELS:
        notes.append(
            "GLMM notes include group structure, variance components, and convergence status."
        )

    return RegressionPublicationReport(
        model_id=regression_result.model_id,
        model_type=regression_result.model_type,
        publication_table=publication_table,
        model_summary=model_summary,
        narrative=narrative,
        notes=notes,
        metadata={
            "dependent_variable": regression_result.dependent_variable,
            "sample_size": regression_result.sample_size,
            "group_variable": regression_result.metadata.get("group_variable"),
            "group_count": regression_result.fit_statistics.get("group_count"),
            "level2_group": regression_result.metadata.get("level2_group"),
            "level3_group": regression_result.metadata.get("level3_group"),
            "level2_group_count": regression_result.fit_statistics.get("level2_group_count"),
            "level3_group_count": regression_result.fit_statistics.get("level3_group_count"),
            "converged": regression_result.converged,
            "optimizer": regression_result.metadata.get("optimizer"),
            "reml": regression_result.metadata.get("reml"),
            "random_effect_covariance": regression_result.metadata.get("random_effect_covariance"),
            "covariance_structure": regression_result.metadata.get("covariance_structure"),
            "binary_link": regression_result.metadata.get("link") if regression_result.model_type in {"binary_cloglog", "binary_probit"} else None,
            "ordered_link": regression_result.metadata.get("link") if regression_result.model_type == "ordered_probit" else None,
            "brier_score": regression_result.fit_statistics.get("brier_score") if regression_result.model_type in {"binary_cloglog", "binary_probit"} else None,
            "reference_category": regression_result.metadata.get("reference_category"),
            "category_labels": regression_result.metadata.get("category_labels"),
            "quantile": regression_result.fit_statistics.get("quantile"),
            "duration_variable": regression_result.metadata.get("duration_variable"),
            "event_variable": regression_result.metadata.get("event_variable"),
            "boundary_count": regression_result.fit_statistics.get("boundary_count"),
            "gamma_dispersion_ratio": regression_result.fit_statistics.get("dispersion_ratio") if regression_result.model_type == "gamma_regression" else None,
            "inverse_gaussian_dispersion_ratio": regression_result.fit_statistics.get("dispersion_ratio") if regression_result.model_type == "inverse_gaussian_regression" else None,
            "precision": regression_result.fit_statistics.get("precision"),
            "iv_endogenous_variables": regression_result.metadata.get("endogenous_variables"),
            "iv_instrument_variables": regression_result.metadata.get("instrument_variables"),
            "minimum_first_stage_f_statistic": regression_result.fit_statistics.get("minimum_first_stage_f_statistic"),
            "selection_variable": regression_result.metadata.get("selection_variable"),
            "selection_rate": regression_result.fit_statistics.get("selection_rate"),
            "inverse_mills_p_value": regression_result.fit_statistics.get("inverse_mills_p_value"),
            "exclusion_restrictions": regression_result.metadata.get("exclusion_restrictions"),
            "regularized_penalty": regression_result.fit_statistics.get("penalty"),
            "regularized_alpha": regression_result.fit_statistics.get("alpha"),
            "regularized_l1_ratio": regression_result.fit_statistics.get("l1_ratio"),
            "selected_coefficient_count": regression_result.fit_statistics.get("selected_coefficient_count"),
            "zero_coefficient_count": regression_result.fit_statistics.get("zero_coefficient_count"),
            "robust_norm": regression_result.metadata.get("norm"),
            "downweighted_count": regression_result.fit_statistics.get("downweighted_count"),
            "heavily_downweighted_count": regression_result.fit_statistics.get("heavily_downweighted_count"),
            "lower_limit": regression_result.metadata.get("lower_limit"),
            "upper_limit": regression_result.metadata.get("upper_limit"),
            "censoring_rate": regression_result.fit_statistics.get("censoring_rate"),
            "entity_variable": regression_result.metadata.get("entity_variable"),
            "time_variable": regression_result.metadata.get("time_variable"),
            "entity_count": regression_result.fit_statistics.get("entity_count"),
            "time_period_count": regression_result.fit_statistics.get("time_period_count"),
            "within_r_squared": regression_result.fit_statistics.get("within_r_squared"),
        },
    )


def publication_table_to_dataframe(
    report: RegressionPublicationReport,
) -> pd.DataFrame:
    return report.publication_table.copy()


def model_summary_to_dataframe(
    report: RegressionPublicationReport,
) -> pd.DataFrame:
    return report.model_summary.copy()
