"""회귀결과를 논문용 표와 한국어 결과문으로 변환한다."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.statistics.regression.base import RegressionResult


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
        elif coefficient.term.lower() in {"const", "intercept"}:
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


def write_korean_results_narrative(
    regression_result: RegressionResult,
    effect_report: Any | None = None,
) -> str:
    """논문 결과 절에 사용할 한국어 서술 초안을 생성한다."""
    effect_lookup = _effect_lookup(effect_report)
    substantive = [
        coefficient
        for coefficient in regression_result.coefficients
        if coefficient.term.lower() not in {"const", "intercept"} and "/" not in coefficient.term
    ]

    sentences: list[str] = []

    model_name = {
        "ols": "OLS 회귀분석",
        "binary_logit": "이항 로지스틱 회귀분석",
        "ordered_logit": "순서형 로지스틱 회귀분석",
        "poisson": "포아송 회귀분석",
        "negative_binomial": "음이항 회귀분석",
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

        if regression_result.model_type == "ols":
            beta = effect_lookup.get(
                (
                    coefficient.term,
                    "standardized_beta",
                )
            )

            if beta is not None:
                effect_text = f"β={beta:.3f}"
            else:
                effect_text = f"B={coefficient.estimate:.3f}"

        elif regression_result.model_type in {
            "binary_logit",
            "ordered_logit",
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
        elif regression_result.model_type in {"poisson", "negative_binomial"}:
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

    elif regression_result.model_type == "binary_logit":
        pseudo = regression_result.fit_statistics.get("pseudo_r_squared_mcfadden")
        lr_p = regression_result.fit_statistics.get("likelihood_ratio_p_value")

        if pseudo is not None:
            sentences.append(f"McFadden 의사 R²는 {pseudo:.3f}이었다.")

        if lr_p is not None:
            sentences.append(f"우도비 검정의 유의확률은 {_format_p_value(float(lr_p))}였다.")

    elif regression_result.model_type == "poisson":
        dispersion_ratio = regression_result.fit_statistics.get("dispersion_ratio")
        pseudo = regression_result.fit_statistics.get("pseudo_r_squared_deviance")

        if pseudo is not None:
            sentences.append(f"Deviance 기반 의사 R²는 {float(pseudo):.3f}이었다.")

        if dispersion_ratio is not None:
            sentences.append(f"Pearson 분산비는 {float(dispersion_ratio):.3f}이었다.")

    elif regression_result.model_type == "negative_binomial":
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

    if regression_result.model_type == "ols":
        notes.append("OLS의 표준화 β와 부분 효과크기를 함께 제시한다.")
    elif regression_result.model_type in {
        "binary_logit",
        "ordered_logit",
    }:
        notes.append("로짓 모형은 오즈비를 함께 제시한다.")
    elif regression_result.model_type in {"poisson", "negative_binomial"}:
        notes.append("계수형 회귀모형은 발생률비(IRR)를 함께 제시한다.")

    return RegressionPublicationReport(
        model_id=regression_result.model_id,
        model_type=regression_result.model_type,
        publication_table=publication_table,
        model_summary=model_summary,
        narrative=narrative,
        notes=notes,
        metadata={
            "dependent_variable": (regression_result.dependent_variable),
            "sample_size": regression_result.sample_size,
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
