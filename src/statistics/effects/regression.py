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


def build_regression_effect_size_report(
    result: RegressionResult,
) -> EffectSizeReport:
    """회귀모형 종류에 맞는 효과크기 보고서를 생성한다."""
    if result.model_type == "ols":
        return _build_ols_effects(result)

    if result.model_type == "binary_logit":
        return _build_binary_logit_effects(result)

    if result.model_type == "ordered_logit":
        return _build_ordered_logit_effects(result)

    if result.model_type in {
        "poisson",
        "negative_binomial",
        "zero_inflated_poisson",
        "zero_inflated_negative_binomial",
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
