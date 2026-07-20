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
        elif coefficient.term.startswith("inflate_"):
            term_type = "inflation"
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
        if coefficient.term.lower() not in {"const", "intercept"}
        and "/" not in coefficient.term
        and not coefficient.term.startswith("inflate_")
    ]

    sentences: list[str] = []

    cross_level = regression_result.metadata.get("cross_level_interaction")

    model_name = {
        "ols": "OLS 회귀분석",
        "binary_logit": "이항 로지스틱 회귀분석",
        "ordered_logit": "순서형 로지스틱 회귀분석",
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

        if regression_result.model_type in {"ols", "mixed_random_intercept", "mixed_random_slope"}:
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
            "mixed_binary_logit_random_intercept",
            "mixed_binary_logit_random_slope",
            "mixed_binary_logit_three_level",
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

    elif regression_result.model_type in {
        "binary_logit",
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

    if regression_result.model_type == "ols":
        notes.append("OLS의 표준화 β와 부분 효과크기를 함께 제시한다.")
    elif regression_result.model_type in {
        "binary_logit",
        "mixed_binary_logit_random_intercept",
        "mixed_binary_logit_random_slope",
        "mixed_binary_logit_three_level",
        "ordered_logit",
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
