"""연구 파이프라인 산출물을 바탕으로 제출 전 품질 감사를 수행한다."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from src.pipeline.runtime import PipelineRuntime

_THREE_LEVEL_MIXED_MODELS = {
    "mixed_three_level",
    "mixed_binary_logit_three_level",
    "mixed_poisson_three_level",
    "mixed_negative_binomial_three_level",
}

_MIXED_RANDOM_SLOPE_MODELS = {
    "mixed_random_slope",
    "mixed_binary_logit_random_slope",
    "mixed_poisson_random_slope",
    "mixed_negative_binomial_random_slope",
}


@dataclass(slots=True)
class AuditItem:
    """개별 연구 품질 점검 항목."""

    category: str
    item: str
    status: str
    score: int
    maximum_score: int
    evidence: str
    recommendation: str


@dataclass(slots=True)
class ResearchAuditReport:
    """연구 품질 감사 전체 결과."""

    model_id: str
    items: list[AuditItem]
    total_score: int
    maximum_score: int
    percentage: float
    grade: str
    submission_status: str
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _artifact_exists(
    runtime: PipelineRuntime,
    key: str,
) -> bool:
    return key in runtime.artifacts


def _regression_result(
    runtime: PipelineRuntime,
    model_id: str,
) -> Any | None:
    return runtime.artifacts.get(f"regression_result:{model_id}")


def _is_mixed_effects_model(
    runtime: PipelineRuntime,
    model_id: str,
) -> bool:
    result = _regression_result(runtime, model_id)
    return bool(
        result is not None
        and getattr(result, "model_type", None)
        in {
            "mixed_random_intercept",
            "mixed_random_slope",
            "mixed_three_level",
            "mixed_binary_logit_random_intercept",
            "mixed_binary_logit_random_slope",
            "mixed_binary_logit_three_level",
            "mixed_poisson_random_intercept",
            "mixed_poisson_random_slope",
            "mixed_poisson_three_level",
            "mixed_negative_binomial_random_intercept",
        "mixed_negative_binomial_random_slope",
        "mixed_negative_binomial_three_level",
        }
    )


def _grade_from_percentage(percentage: float) -> str:
    if percentage >= 90:
        return "A"
    if percentage >= 80:
        return "B"
    if percentage >= 70:
        return "C"
    if percentage >= 60:
        return "D"
    return "F"


def _submission_status(percentage: float) -> str:
    if percentage >= 90:
        return "제출 준비 완료"
    if percentage >= 80:
        return "경미한 보완 후 제출 가능"
    if percentage >= 70:
        return "주요 보완 필요"
    return "제출 전 재검토 필요"


def _missingness_item(runtime: PipelineRuntime) -> AuditItem:
    report = getattr(runtime, "missingness_report", None)

    if report is None:
        return AuditItem(
            category="데이터 품질",
            item="결측치 진단",
            status="MISSING",
            score=0,
            maximum_score=10,
            evidence="결측치 진단 결과가 없습니다.",
            recommendation="결측치 패턴과 처리방식을 먼저 확인하세요.",
        )

    return AuditItem(
        category="데이터 품질",
        item="결측치 진단",
        status="PASS",
        score=10,
        maximum_score=10,
        evidence="결측치 진단 결과가 생성되었습니다.",
        recommendation="논문 연구방법에 결측치 처리기준을 명시하세요.",
    )


def _outlier_item(runtime: PipelineRuntime) -> AuditItem:
    report = getattr(runtime, "outlier_report", None)

    if report is None:
        return AuditItem(
            category="데이터 품질",
            item="이상치 진단",
            status="MISSING",
            score=0,
            maximum_score=10,
            evidence="이상치 진단 결과가 없습니다.",
            recommendation="단변량·다변량 이상치를 점검하세요.",
        )

    return AuditItem(
        category="데이터 품질",
        item="이상치 진단",
        status="PASS",
        score=10,
        maximum_score=10,
        evidence="이상치 진단 결과가 생성되었습니다.",
        recommendation="제외 여부와 민감도 분석 결과를 함께 기록하세요.",
    )


def _reliability_item(runtime: PipelineRuntime) -> AuditItem:
    candidates = [
        "scale_reliability_report",
        "reliability_report",
    ]

    exists = any(key in runtime.artifacts for key in candidates)

    if exists:
        return AuditItem(
            category="측정 품질",
            item="척도 신뢰도",
            status="PASS",
            score=10,
            maximum_score=10,
            evidence="척도 신뢰도 결과가 저장되어 있습니다.",
            recommendation="Cronbach's α 또는 적절한 신뢰도 지표를 보고하세요.",
        )

    return AuditItem(
        category="측정 품질",
        item="척도 신뢰도",
        status="REVIEW",
        score=5,
        maximum_score=10,
        evidence="신뢰도 산출물을 확인하지 못했습니다.",
        recommendation="복수문항 척도가 있다면 신뢰도를 보고하세요.",
    )


def _regression_item(
    runtime: PipelineRuntime,
    model_id: str,
) -> AuditItem:
    key = f"regression_result:{model_id}"

    if not _artifact_exists(runtime, key):
        return AuditItem(
            category="모형 분석",
            item="회귀모형 추정",
            status="MISSING",
            score=0,
            maximum_score=15,
            evidence="회귀분석 결과가 없습니다.",
            recommendation="연구가설에 맞는 회귀모형을 실행하세요.",
        )

    result = runtime.artifacts[key]
    status = "PASS" if result.converged else "FAIL"
    score = 15 if result.converged else 5
    selected = result.metadata.get("selected_survival_model")
    selection_suffix = ""
    if selected:
        selection_suffix = (
            f", selected survival model={selected}, "
            f"criterion={result.metadata.get('survival_selection_criterion', 'aic')}"
        )





    if result.model_type == "exponential_aft":
        evidence = (
            f"Exponential AFT regression, N={result.sample_size}, "
            f"events={result.fit_statistics.get('event_count', 'unknown')}, "
            f"censored={result.fit_statistics.get('censored_count', 'unknown')}, "
            f"constant hazard={result.fit_statistics.get('constant_hazard', 'unknown')}, "
            f"converged={result.converged}{selection_suffix}"
        )
        return AuditItem(
            category="?? ??",
            item="???? ??",
            status=status,
            score=score,
            maximum_score=15,
            evidence=evidence,
            recommendation="Report duration/event coding, exponential distribution, censoring, constant-hazard assumption, and time ratios.",
        )

    if result.model_type == "loglogistic_aft":
        evidence = (
            f"Log-logistic AFT regression, N={result.sample_size}, "
            f"events={result.fit_statistics.get('event_count', 'unknown')}, "
            f"censored={result.fit_statistics.get('censored_count', 'unknown')}, "
            f"shape={result.fit_statistics.get('shape', 'unknown')}, "
            f"converged={result.converged}"
        )
        return AuditItem(
            category="?? ??",
            item="???? ??",
            status=status,
            score=score,
            maximum_score=15,
            evidence=evidence,
            recommendation="Report duration/event coding, log-logistic distribution, censoring, time ratios, and convergence.",
        )

    if result.model_type == "lognormal_aft":
        evidence = (
            f"Log-normal AFT regression, N={result.sample_size}, "
            f"events={result.fit_statistics.get('event_count', 'unknown')}, "
            f"censored={result.fit_statistics.get('censored_count', 'unknown')}, "
            f"sigma={result.fit_statistics.get('sigma', 'unknown')}, "
            f"converged={result.converged}"
        )
        return AuditItem(
            category="?? ??",
            item="???? ??",
            status=status,
            score=score,
            maximum_score=15,
            evidence=evidence,
            recommendation="Report duration/event coding, log-normal distribution, censoring, time ratios, and convergence.",
        )

    if result.model_type == "weibull_ph":
        evidence = (
            f"Weibull PH regression, N={result.sample_size}, "
            f"events={result.fit_statistics.get('event_count', 'unknown')}, "
            f"censored={result.fit_statistics.get('censored_count', 'unknown')}, "
            f"shape={result.fit_statistics.get('shape', 'unknown')}, "
            f"converged={result.converged}"
        )
        return AuditItem(
            category="?? ??",
            item="???? ??",
            status=status,
            score=score,
            maximum_score=15,
            evidence=evidence,
            recommendation="Report duration/event coding, Weibull PH parameterization, censoring, hazard ratios, and convergence.",
        )

    if result.model_type == "weibull_aft":
        evidence = (
            f"Weibull AFT regression, N={result.sample_size}, "
            f"events={result.fit_statistics.get('event_count', 'unknown')}, "
            f"censored={result.fit_statistics.get('censored_count', 'unknown')}, "
            f"shape={result.fit_statistics.get('shape', 'unknown')}, "
            f"converged={result.converged}"
        )
        return AuditItem(
            category="?? ??",
            item="???? ??",
            status=status,
            score=score,
            maximum_score=15,
            evidence=evidence,
            recommendation="Report duration/event coding, Weibull distribution, censoring, time ratios, and convergence.",
        )

    if result.model_type == "piecewise_exponential":
        evidence = (
            f"Piecewise exponential regression, N={result.sample_size}, "
            f"events={result.fit_statistics.get('event_count', 'unknown')}, "
            f"intervals={result.fit_statistics.get('interval_count', 'unknown')}, "
            f"total exposure={result.fit_statistics.get('total_exposure', 'unknown')}, "
            f"converged={result.converged}"
        )
        return AuditItem(
            category="?? ??",
            item="???? ??",
            status=status,
            score=score,
            maximum_score=15,
            evidence=evidence,
            recommendation="Report interval cut points, duration/event coding, exposure offset, baseline hazards, and hazard ratios.",
        )

    if result.model_type == "discrete_time_hazard":
        evidence = (
            f"Discrete-time hazard regression, N={result.sample_size}, "
            f"events={result.fit_statistics.get('event_count', 'unknown')}, "
            f"intervals={result.fit_statistics.get('interval_count', 'unknown')}, "
            f"link={result.metadata.get('link', 'unknown')}, "
            f"converged={result.converged}"
        )
        return AuditItem(
            category="?? ??",
            item="???? ??",
            status=status,
            score=score,
            maximum_score=15,
            evidence=evidence,
            recommendation="Report interval cut points, duration/event coding, person-period construction, link function, and hazard effects.",
        )

    if result.model_type == "log_binomial":
        evidence = (
            f"Log-binomial regression, N={result.sample_size}, "
            f"events={result.fit_statistics.get('event_count', 'unknown')}, "
            f"non-events={result.fit_statistics.get('non_event_count', 'unknown')}, "
            f"Brier={result.fit_statistics.get('brier_score', 'unknown')}, "
            f"out-of-bounds predictions={result.fit_statistics.get('out_of_bounds_prediction_count', 'unknown')}, "
            f"converged={result.converged}"
        )
        return AuditItem(
            category="?? ??",
            item="???? ??",
            status=status,
            score=score,
            maximum_score=15,
            evidence=evidence,
            recommendation="Report the log link, event coding, risk ratios, fitted-probability bounds, and marginal effects.",
        )

    if result.model_type == "weighted_least_squares":
        evidence = (
            f"Weighted least squares regression, N={result.sample_size}, "
            f"weight={result.metadata.get('weight_variable', 'unknown')}, "
            f"R-squared={result.fit_statistics.get('r_squared', 'unknown')}, "
            f"weight ratio={result.fit_statistics.get('weight_ratio', 'unknown')}, "
            f"converged={result.converged}"
        )
        return AuditItem(
            category="?? ??",
            item="???? ??",
            status=status,
            score=score,
            maximum_score=15,
            evidence=evidence,
            recommendation="Report the weight variable, weight construction rationale, weight range, and robust standard errors.",
        )

    if result.model_type == "binary_cloglog":
        evidence = (
            f"Binary cloglog regression, N={result.sample_size}, "
            f"events={result.fit_statistics.get('event_count', 'unknown')}, "
            f"non-events={result.fit_statistics.get('non_event_count', 'unknown')}, "
            f"Brier={result.fit_statistics.get('brier_score', 'unknown')}, "
            f"converged={result.converged}"
        )
        return AuditItem(
            category="?? ??",
            item="???? ??",
            status=status,
            score=score,
            maximum_score=15,
            evidence=evidence,
            recommendation="Report the complementary log-log link, event coding, classification diagnostics, and marginal effects.",
        )

    if result.model_type == "ordered_probit":
        evidence = (
            f"Ordered probit regression, N={result.sample_size}, "
            f"categories={result.fit_statistics.get('category_count', 'unknown')}, "
            f"converged={result.converged}"
        )
        return AuditItem(
            category="?? ??",
            item="???? ??",
            status=status,
            score=score,
            maximum_score=15,
            evidence=evidence,
            recommendation="Report the probit link, ordinal category coding, threshold parameters, and prediction diagnostics.",
        )

    if result.model_type == "binary_probit":
        evidence = (
            f"Binary probit regression, N={result.sample_size}, "
            f"events={result.fit_statistics.get('event_count', 'unknown')}, "
            f"non-events={result.fit_statistics.get('non_event_count', 'unknown')}, "
            f"pseudo R-squared={result.fit_statistics.get('pseudo_r_squared_mcfadden', 'unknown')}, "
            f"converged={result.converged}"
        )
        return AuditItem(
            category="?? ??",
            item="???? ??",
            status=status,
            score=score,
            maximum_score=15,
            evidence=evidence,
            recommendation="Report the probit link, event coding, classification diagnostics, and average marginal effects.",
        )

    if result.model_type == "heckman_selection":
        selection_variable = result.metadata.get("selection_variable", "unknown")
        selection_rate = result.fit_statistics.get("selection_rate", "unknown")
        imr_p = result.fit_statistics.get("inverse_mills_p_value", "unknown")
        exclusions = result.metadata.get("exclusion_restrictions", [])
        evidence = (
            f"Heckman selection regression, N={result.sample_size}, "
            f"selection variable={selection_variable}, selection rate={selection_rate}, "
            f"exclusions={exclusions}, inverse Mills p={imr_p}, converged={result.converged}"
        )
        return AuditItem(
            category="?? ??",
            item="???? ??",
            status=status,
            score=score,
            maximum_score=15,
            evidence=evidence,
            recommendation="Report the selection equation, exclusion restrictions, selection rate, and inverse Mills ratio test.",
        )

    if result.model_type == "iv_2sls_regression":
        endogenous = result.metadata.get("endogenous_variables", [])
        instruments = result.metadata.get("instrument_variables", [])
        min_f = result.fit_statistics.get("minimum_first_stage_f_statistic", "unknown")
        evidence = (
            f"IV 2SLS regression, N={result.sample_size}, "
            f"endogenous={endogenous}, instruments={instruments}, "
            f"minimum first-stage F={min_f}, converged={result.converged}"
        )
        return AuditItem(
            category="?? ??",
            item="???? ??",
            status=status,
            score=score,
            maximum_score=15,
            evidence=evidence,
            recommendation="Report endogenous variables, excluded instruments, first-stage strength, and exclusion rationale.",
        )

    if result.model_type == "inverse_gaussian_regression":
        dispersion = result.fit_statistics.get("dispersion_ratio", "unknown")
        pseudo = result.fit_statistics.get("pseudo_r_squared_deviance", "unknown")
        evidence = (
            f"Inverse Gaussian regression, N={result.sample_size}, "
            f"dispersion={dispersion}, deviance pseudo R-squared={pseudo}, "
            f"converged={result.converged}"
        )
        return AuditItem(
            category="?? ??",
            item="???? ??",
            status=status,
            score=score,
            maximum_score=15,
            evidence=evidence,
            recommendation="Report Inverse Gaussian log-link rationale, positive outcome support, dispersion, and mean ratios.",
        )

    if result.model_type == "gamma_regression":
        dispersion = result.fit_statistics.get("dispersion_ratio", "unknown")
        pseudo = result.fit_statistics.get("pseudo_r_squared_deviance", "unknown")
        evidence = (
            f"Gamma regression, N={result.sample_size}, "
            f"dispersion={dispersion}, deviance pseudo R-squared={pseudo}, "
            f"converged={result.converged}"
        )
        return AuditItem(
            category="?? ??",
            item="???? ??",
            status=status,
            score=score,
            maximum_score=15,
            evidence=evidence,
            recommendation="Report Gamma log-link rationale, positive outcome support, dispersion, and mean ratios.",
        )

    if result.model_type == "regularized_regression":
        penalty = result.fit_statistics.get("penalty", "unknown")
        alpha = result.fit_statistics.get("alpha", "unknown")
        selected = result.fit_statistics.get("selected_coefficient_count", "unknown")
        zero = result.fit_statistics.get("zero_coefficient_count", "unknown")
        evidence = (
            f"Regularized linear regression, N={result.sample_size}, "
            f"penalty={penalty}, alpha={alpha}, selected={selected}, zero={zero}, "
            f"converged={result.converged}"
        )
        return AuditItem(
            category="?? ??",
            item="???? ??",
            status=status,
            score=score,
            maximum_score=15,
            evidence=evidence,
            recommendation="Report the penalty type, alpha/l1_ratio, selected coefficients, and tuning rationale.",
        )

    if result.model_type == "robust_regression":
        downweighted = result.fit_statistics.get("downweighted_count", "unknown")
        heavy = result.fit_statistics.get("heavily_downweighted_count", "unknown")
        evidence = (
            f"Robust linear regression, N={result.sample_size}, "
            f"downweighted={downweighted}, heavily downweighted={heavy}, "
            f"converged={result.converged}"
        )
        return AuditItem(
            category="?? ??",
            item="???? ??",
            status=status,
            score=score,
            maximum_score=15,
            evidence=evidence,
            recommendation="Report the robust norm, downweighted cases, and why robust estimation was selected.",
        )

    if result.model_type == "tobit_regression":
        left_count = result.fit_statistics.get("left_censored_count", "unknown")
        right_count = result.fit_statistics.get("right_censored_count", "unknown")
        censoring_rate = result.fit_statistics.get("censoring_rate")
        evidence = (
            f"Tobit censored regression, N={result.sample_size}, "
            f"left-censored={left_count}, right-censored={right_count}, "
            f"converged={result.converged}"
        )
        if censoring_rate is not None:
            evidence += f", censoring rate={float(censoring_rate):.3f}"
        return AuditItem(
            category="?? ??",
            item="???? ??",
            status=status,
            score=score,
            maximum_score=15,
            evidence=evidence,
            recommendation="Report Tobit censoring limits, censored counts, and latent-scale estimation method.",
        )

    if result.model_type in {"panel_fixed_effects", "panel_random_effects", "panel_correlated_random_effects", "panel_between_effects", "panel_first_difference", "panel_pooled_ols"}:
        entity_variable = result.metadata.get("entity_variable", "unknown")
        time_variable = result.metadata.get("time_variable")
        entity_count = result.fit_statistics.get("entity_count", "unknown")
        time_count = result.fit_statistics.get("time_period_count")
        within_r_squared = result.fit_statistics.get("within_r_squared", result.fit_statistics.get("marginal_r_squared", result.fit_statistics.get("between_r_squared", result.fit_statistics.get("first_difference_r_squared", result.fit_statistics.get("pooled_r_squared")))))
        evidence = (
            f"Panel model, N={result.sample_size}, "
            f"entity={entity_variable}({entity_count}), converged={result.converged}"
        )
        if time_variable is not None:
            evidence += f", time={time_variable}({time_count})"
        if within_r_squared is not None:
            evidence += f", panel R-squared={float(within_r_squared):.3f}"
        return AuditItem(
            category="?? ??",
            item="???? ??",
            status=status,
            score=score,
            maximum_score=15,
            evidence=evidence,
            recommendation="Report absorbed entity/time effects, within-panel estimates, and clustered or robust standard errors.",
        )

    if result.model_type in {
        "mixed_random_intercept",
        "mixed_random_slope",
        "mixed_three_level",
        "mixed_binary_logit_random_intercept",
        "mixed_binary_logit_random_slope",
        "mixed_binary_logit_three_level",
        "mixed_poisson_random_intercept",
        "mixed_poisson_random_slope",
        "mixed_poisson_three_level",
        "mixed_negative_binomial_random_intercept",
        "mixed_negative_binomial_random_slope",
        "mixed_negative_binomial_three_level",
    }:
        group_variable = result.metadata.get("group_variable", "미확인")
        group_count = result.fit_statistics.get("group_count", "미확인")
        cross_level = result.metadata.get("cross_level_interaction")
        if result.model_type in _THREE_LEVEL_MIXED_MODELS:
            level2_group = result.metadata.get("level2_group", "미확인")
            level3_group = result.metadata.get("level3_group", "미확인")
            level2_count = result.fit_statistics.get("level2_group_count", "미확인")
            level3_count = result.fit_statistics.get("level3_group_count", "미확인")
            structure = "3수준 중첩 Random Effects"
            evidence = (
                f"{structure} 모형, N={result.sample_size}, "
                f"Level 2={level2_group}({level2_count}개), "
                f"Level 3={level3_group}({level3_count}개), 수렴={result.converged}"
            )
            recommendation = (
                "Level 1·2·3의 단위, 완전 중첩 구조, 수준별 분산과 ICC, "
                "랜덤효과 설정 및 추정방법을 연구방법과 결과에 명시하세요."
            )
            return AuditItem(
                category="모형 분석",
                item="회귀모형 추정",
                status=status,
                score=score,
                maximum_score=15,
                evidence=evidence,
                recommendation=recommendation,
            )
        structure = (
            "3-Level"
            if result.model_type in _THREE_LEVEL_MIXED_MODELS
            else "Random Slope"
            if result.model_type in _MIXED_RANDOM_SLOPE_MODELS
            else "Random Intercept"
        )
        covariance_structure = result.metadata.get("random_effect_covariance", "correlated")
        covariance_text = "비상관(대각)" if covariance_structure == "diagonal" else "상관"
        evidence = (
            f"{structure} 모형, N={result.sample_size}, "
            f"그룹변수={group_variable}, 그룹 수={group_count}, "
            f"랜덤효과 공분산={covariance_text}, 수렴={result.converged}"
        )
        if cross_level:
            evidence += (
                f", 교차수준 상호작용={cross_level.get('predictor')}×{cross_level.get('moderator')}, "
                f"중심화={cross_level.get('level1_centering')}/{cross_level.get('level2_centering')}"
            )
            recommendation = "교차수준 상호작용, 중심화 방식, Random Slope 구조와 조건부 효과를 연구방법 및 결과에 명시하세요."
        else:
            recommendation = (
                "그룹 구조, 고정효과, 랜덤효과 설정과 추정방법을 연구방법에 명시하세요."
            )
    else:
        evidence = f"{result.model_type} 모형, N={result.sample_size}, 수렴={result.converged}"
        recommendation = "모형 선택근거와 표준오차 추정방식을 명시하세요."

    return AuditItem(
        category="모형 분석",
        item="회귀모형 추정",
        status=status,
        score=score,
        maximum_score=15,
        evidence=evidence,
        recommendation=recommendation,
    )


def _diagnostics_item(
    runtime: PipelineRuntime,
    model_id: str,
) -> AuditItem:
    key = f"regression_diagnostics:{model_id}"

    if not _artifact_exists(runtime, key):
        return AuditItem(
            category="모형 검증",
            item="회귀진단",
            status="REVIEW",
            score=5,
            maximum_score=10,
            evidence="OLS 진단 결과가 없거나 비OLS 모형입니다.",
            recommendation="모형 유형에 적절한 진단검정을 추가하세요.",
        )

    report = runtime.artifacts[key]
    warning_count = len(report.warnings)

    result = _regression_result(runtime, model_id)




    if result is not None and getattr(result, "model_type", None) == "exponential_aft":
        summary = getattr(report, "summary", {})
        warning_count = len(getattr(report, "warnings", []))
        evidence = (
            f"Exponential AFT diagnostics, C-index={summary.get('concordance_index', 'unknown')}, "
            f"events per parameter={summary.get('events_per_parameter', 'unknown')}, "
            f"warnings={warning_count}"
        )
        return AuditItem(
            category="?? ??",
            item="?? ??",
            status="PASS" if warning_count == 0 else "WARNING",
            score=10 if warning_count == 0 else 7,
            maximum_score=10,
            evidence=evidence,
            recommendation="Report AFT residuals, concordance, VIF screening, censoring, and constant-hazard assumption.",
        )

    if result is not None and getattr(result, "model_type", None) == "loglogistic_aft":
        summary = getattr(report, "summary", {})
        warning_count = len(getattr(report, "warnings", []))
        evidence = (
            f"Log-logistic AFT diagnostics, C-index={summary.get('concordance_index', 'unknown')}, "
            f"events per parameter={summary.get('events_per_parameter', 'unknown')}, "
            f"warnings={warning_count}"
        )
        return AuditItem(
            category="?? ??",
            item="?? ??",
            status="PASS" if warning_count == 0 else "WARNING",
            score=10 if warning_count == 0 else 7,
            maximum_score=10,
            evidence=evidence,
            recommendation="Report AFT residuals, concordance, VIF screening, censoring, and events per parameter.",
        )

    if result is not None and getattr(result, "model_type", None) == "lognormal_aft":
        summary = getattr(report, "summary", {})
        warning_count = len(getattr(report, "warnings", []))
        evidence = (
            f"Log-normal AFT diagnostics, C-index={summary.get('concordance_index', 'unknown')}, "
            f"events per parameter={summary.get('events_per_parameter', 'unknown')}, "
            f"warnings={warning_count}"
        )
        return AuditItem(
            category="?? ??",
            item="?? ??",
            status="PASS" if warning_count == 0 else "WARNING",
            score=10 if warning_count == 0 else 7,
            maximum_score=10,
            evidence=evidence,
            recommendation="Report AFT residuals, concordance, VIF screening, censoring, and events per parameter.",
        )

    if result is not None and getattr(result, "model_type", None) == "weibull_aft":
        summary = getattr(report, "summary", {})
        warning_count = len(getattr(report, "warnings", []))
        evidence = (
            f"Weibull AFT diagnostics, C-index={summary.get('concordance_index', 'unknown')}, "
            f"events per parameter={summary.get('events_per_parameter', 'unknown')}, "
            f"warnings={warning_count}"
        )
        return AuditItem(
            category="?? ??",
            item="?? ??",
            status="PASS" if warning_count == 0 else "WARNING",
            score=10 if warning_count == 0 else 7,
            maximum_score=10,
            evidence=evidence,
            recommendation="Report AFT residuals, concordance, VIF screening, censoring, and events per parameter.",
        )

    if result is not None and getattr(result, "model_type", None) == "log_binomial":
        summary = getattr(report, "summary", {})
        warning_count = len(getattr(report, "warnings", []))
        evidence = (
            f"Log-binomial diagnostics, ROC-AUC={summary.get('roc_auc', 'unknown')}, "
            f"Brier={summary.get('brier_score', 'unknown')}, warnings={warning_count}"
        )
        return AuditItem(
            category="?? ??",
            item="?? ??",
            status="PASS" if warning_count == 0 else "WARNING",
            score=10 if warning_count == 0 else 7,
            maximum_score=10,
            evidence=evidence,
            recommendation="Report ROC-AUC, Brier score, calibration, VIF screening, and probability-bound warnings.",
        )

    if result is not None and getattr(result, "model_type", None) == "weighted_least_squares":
        warning_count = len(getattr(report, "warnings", []))
        evidence = f"WLS diagnostics, diagnostic warnings={warning_count}"
        return AuditItem(
            category="?? ??",
            item="?? ??",
            status="PASS" if warning_count == 0 else "WARNING",
            score=10 if warning_count == 0 else 7,
            maximum_score=10,
            evidence=evidence,
            recommendation="Report WLS residual diagnostics, heteroskedasticity tests, VIF screening, and influence checks.",
        )

    if result is not None and getattr(result, "model_type", None) == "binary_cloglog":
        summary = getattr(report, "summary", {})
        warning_count = len(getattr(report, "warnings", []))
        evidence = (
            f"Binary cloglog diagnostics, ROC-AUC={summary.get('roc_auc', 'unknown')}, "
            f"Brier={summary.get('brier_score', 'unknown')}, warnings={warning_count}"
        )
        return AuditItem(
            category="?? ??",
            item="?? ??",
            status="PASS" if warning_count == 0 else "WARNING",
            score=10 if warning_count == 0 else 7,
            maximum_score=10,
            evidence=evidence,
            recommendation="Report ROC-AUC, Brier score, calibration, and VIF screening for the binary cloglog model.",
        )

    if result is not None and getattr(result, "model_type", None) == "ordered_probit":
        summary = getattr(report, "summary", {})
        warning_count = len(getattr(report, "warnings", []))
        evidence = (
            f"Ordered probit diagnostics, accuracy={summary.get('accuracy', 'unknown')}, "
            f"ranked probability score={summary.get('ranked_probability_score', 'unknown')}, "
            f"warnings={warning_count}"
        )
        return AuditItem(
            category="?? ??",
            item="?? ??",
            status="PASS" if warning_count == 0 else "WARNING",
            score=10 if warning_count == 0 else 7,
            maximum_score=10,
            evidence=evidence,
            recommendation="Report ordered-category prediction accuracy, ranked probability score, thresholds, and VIF screening.",
        )

    if result is not None and getattr(result, "model_type", None) == "binary_probit":
        summary = getattr(report, "summary", {})
        warning_count = len(getattr(report, "warnings", []))
        evidence = (
            f"Binary probit diagnostics, ROC-AUC={summary.get('roc_auc', 'unknown')}, "
            f"Brier={summary.get('brier_score', 'unknown')}, warnings={warning_count}"
        )
        return AuditItem(
            category="?? ??",
            item="?? ??",
            status="PASS" if warning_count == 0 else "WARNING",
            score=10 if warning_count == 0 else 7,
            maximum_score=10,
            evidence=evidence,
            recommendation="Report ROC-AUC, Brier score, calibration, and VIF screening for the binary probit model.",
        )

    if result is not None and getattr(result, "model_type", None) == "heckman_selection":
        summary = getattr(report, "summary", {})
        warning_count = len(getattr(report, "warnings", []))
        evidence = (
            f"Heckman diagnostics, selection rate={summary.get('selection_rate', 'unknown')}, "
            f"inverse Mills p={summary.get('inverse_mills_p_value', 'unknown')}, "
            f"exclusions={summary.get('exclusion_restriction_count', 'unknown')}, "
            f"diagnostic warnings={warning_count}"
        )
        return AuditItem(
            category="?? ??",
            item="?? ??",
            status="PASS" if warning_count == 0 else "WARNING",
            score=10 if warning_count == 0 else 7,
            maximum_score=10,
            evidence=evidence,
            recommendation="Report the first-stage selection diagnostics, inverse Mills ratio, residual checks, and VIF screening.",
        )

    if result is not None and getattr(result, "model_type", None) == "iv_2sls_regression":
        summary = getattr(report, "summary", {})
        warning_count = len(getattr(report, "warnings", []))
        evidence = (
            f"IV diagnostics, minimum first-stage F={summary.get('minimum_first_stage_f_statistic', 'unknown')}, "
            f"weak-instrument warning={summary.get('weak_instrument_warning', 'unknown')}, "
            f"diagnostic warnings={warning_count}"
        )
        return AuditItem(
            category="?? ??",
            item="?? ??",
            status="PASS" if warning_count == 0 else "WARNING",
            score=10 if warning_count == 0 else 7,
            maximum_score=10,
            evidence=evidence,
            recommendation="Report first-stage diagnostics, weak-instrument screening, and second-stage residual checks.",
        )

    if result is not None and getattr(result, "model_type", None) == "inverse_gaussian_regression":
        summary = getattr(report, "summary", {})
        warning_count = len(getattr(report, "warnings", []))
        evidence = (
            f"Inverse Gaussian diagnostics, dispersion={summary.get('dispersion_ratio', 'unknown')}, "
            f"RMSE={summary.get('root_mean_squared_error', 'unknown')}, "
            f"diagnostic warnings={warning_count}"
        )
        return AuditItem(
            category="?? ??",
            item="?? ??",
            status="PASS" if warning_count == 0 else "WARNING",
            score=10 if warning_count == 0 else 7,
            maximum_score=10,
            evidence=evidence,
            recommendation="Report Inverse Gaussian prediction diagnostics, dispersion, and VIF screening.",
        )

    if result is not None and getattr(result, "model_type", None) == "gamma_regression":
        summary = getattr(report, "summary", {})
        warning_count = len(getattr(report, "warnings", []))
        evidence = (
            f"Gamma diagnostics, dispersion={summary.get('dispersion_ratio', 'unknown')}, "
            f"RMSE={summary.get('root_mean_squared_error', 'unknown')}, "
            f"diagnostic warnings={warning_count}"
        )
        return AuditItem(
            category="?? ??",
            item="?? ??",
            status="PASS" if warning_count == 0 else "WARNING",
            score=10 if warning_count == 0 else 7,
            maximum_score=10,
            evidence=evidence,
            recommendation="Report Gamma prediction diagnostics, dispersion, and VIF screening.",
        )

    if result is not None and getattr(result, "model_type", None) == "regularized_regression":
        summary = getattr(report, "summary", {})
        warning_count = len(getattr(report, "warnings", []))
        evidence = (
            f"Regularized diagnostics, selected={summary.get('selected_coefficient_count', 'unknown')}, "
            f"zero={summary.get('zero_coefficient_count', 'unknown')}, "
            f"RMSE={summary.get('root_mean_squared_error', 'unknown')}, "
            f"diagnostic warnings={warning_count}"
        )
        return AuditItem(
            category="?? ??",
            item="?? ??",
            status="PASS" if warning_count == 0 else "WARNING",
            score=10 if warning_count == 0 else 7,
            maximum_score=10,
            evidence=evidence,
            recommendation="Report coefficient selection, prediction diagnostics, and VIF screening.",
        )

    if result is not None and getattr(result, "model_type", None) == "robust_regression":
        summary = getattr(report, "summary", {})
        warning_count = len(getattr(report, "warnings", []))
        evidence = (
            f"Robust diagnostics, downweighted={summary.get('downweighted_count', 'unknown')}, "
            f"minimum weight={summary.get('minimum_weight', 'unknown')}, "
            f"diagnostic warnings={warning_count}"
        )
        return AuditItem(
            category="?? ??",
            item="?? ??",
            status="PASS" if warning_count == 0 else "WARNING",
            score=10 if warning_count == 0 else 7,
            maximum_score=10,
            evidence=evidence,
            recommendation="Report robust weights, residual checks, and VIF screening.",
        )

    if result is not None and getattr(result, "model_type", None) == "tobit_regression":
        summary = getattr(report, "summary", {})
        warning_count = len(getattr(report, "warnings", []))
        evidence = (
            f"Tobit diagnostics, censoring rate={summary.get('censoring_rate', 'unknown')}, "
            f"RMSE={summary.get('root_mean_squared_error', 'unknown')}, "
            f"diagnostic warnings={warning_count}"
        )
        return AuditItem(
            category="?? ??",
            item="?? ??",
            status="PASS" if warning_count == 0 else "WARNING",
            score=10 if warning_count == 0 else 7,
            maximum_score=10,
            evidence=evidence,
            recommendation="Report Tobit residual checks, censoring diagnostics, and VIF screening.",
        )

    if result is not None and getattr(result, "model_type", None) == "panel_fixed_effects":
        summary = getattr(report, "summary", {})
        warning_count = len(getattr(report, "warnings", []))
        evidence = (
            f"Panel diagnostics, entities={summary.get('entity_count', 'unknown')}, "
            f"panel R-squared={summary.get('within_r_squared', 'unknown')}, "
            f"diagnostic warnings={warning_count}"
        )
        return AuditItem(
            category="?? ??",
            item="?? ??",
            status="PASS" if warning_count == 0 else "WARNING",
            score=10 if warning_count == 0 else 7,
            maximum_score=10,
            evidence=evidence,
            recommendation="Report within-panel VIF checks and entity-level residual diagnostics.",
        )

    if result is not None and getattr(result, "model_type", None) in {"gee_gaussian", "gee_logit", "gee_poisson", "gee_negative_binomial", "gee_gamma", "gee_inverse_gaussian", "gee_tweedie"}:
        summary = getattr(report, "summary", {})
        warning_count = len(getattr(report, "warnings", []))
        evidence = (
            f"GEE diagnostics, clusters={summary.get('cluster_count', 'unknown')}, "
            f"max cluster mean Pearson residual={summary.get('max_abs_cluster_mean_pearson_residual', 'unknown')}, "
            f"diagnostic warnings={warning_count}"
        )
        return AuditItem(
            category="?? ??",
            item="????",
            status="PASS" if warning_count == 0 else "WARNING",
            score=10 if warning_count == 0 else 7,
            maximum_score=10,
            evidence=evidence,
            recommendation="GEE cluster residual diagnostics and working correlation should be reported.",
        )

    if _is_mixed_effects_model(runtime, model_id):
        summary = getattr(report, "summary", {})
        group_count = getattr(report, "group_count", None)
        singular_fit = bool(summary.get("singular_fit", False))
        converged = bool(summary.get("converged", True))
        evidence = (
            f"혼합효과 진단, 그룹 수={group_count}, 수렴={converged}, "
            f"특이 적합={singular_fit}, 진단 경고 {warning_count}건"
        )
        recommendation = (
            "조건부 잔차, Random Intercept 분포, 특이 적합과 수렴 상태를 함께 보고하세요."
        )
    else:
        evidence = f"진단 경고 {warning_count}건"
        recommendation = "경고 항목이 있다면 강건성 분석과 민감도 분석으로 보완하세요."

    return AuditItem(
        category="모형 검증",
        item="회귀진단",
        status="PASS" if warning_count == 0 else "WARNING",
        score=10 if warning_count == 0 else 7,
        maximum_score=10,
        evidence=evidence,
        recommendation=recommendation,
    )


def _panel_hausman_item(
    runtime: PipelineRuntime,
    model_id: str,
) -> AuditItem | None:
    key = f"panel_hausman:{model_id}"
    if not _artifact_exists(runtime, key):
        return None

    report = runtime.artifacts[key]
    warning_count = len(getattr(report, "warnings", []))
    status = getattr(report, "status", "WARNING")
    if warning_count and status == "PASS":
        status = "WARNING"
    score = 8 if status == "PASS" else 6
    if warning_count:
        score = min(score, 6)
    evidence = (
        f"Panel Hausman diagnostic, FE={getattr(report, 'fixed_model_id', 'unknown')}, "
        f"RE={getattr(report, 'random_model_id', 'unknown')}, "
        f"chi-square={getattr(report, 'statistic', 'unknown')}, "
        f"df={getattr(report, 'degrees_of_freedom', 'unknown')}, "
        f"p-value={getattr(report, 'p_value', 'unknown')}, warnings={warning_count}"
    )
    return AuditItem(
        category="Model diagnostics",
        item="Panel Hausman comparison",
        status=status,
        score=score,
        maximum_score=8,
        evidence=evidence,
        recommendation=getattr(
            report,
            "recommendation",
            "Report FE/RE Hausman comparison when choosing between panel estimators.",
        ),
    )


def _robustness_item(
    runtime: PipelineRuntime,
    model_id: str,
) -> AuditItem:
    if _is_mixed_effects_model(runtime, model_id):
        basic_key = f"robustness_report:{model_id}"
        advanced_key = f"advanced_robustness_report:{model_id}"
        basic_exists = _artifact_exists(runtime, basic_key)
        advanced_exists = _artifact_exists(runtime, advanced_key)
        if basic_exists or advanced_exists:
            methods: list[str] = []
            warnings: list[str] = []
            if basic_exists:
                report = runtime.artifacts[basic_key]
                summary = getattr(report, "summary", {})
                methods.append(
                    "optimizer sensitivity "
                    f"successful refits={summary.get('successful_optimizer_count', 'unknown')}, "
                    f"stable terms={summary.get('stable_term_count', 'unknown')}"
                )
                warnings.extend(getattr(report, "warnings", []))
            if advanced_exists:
                report = runtime.artifacts[advanced_key]
                metadata = getattr(report, "metadata", {})
                methods.append(
                    "group bootstrap "
                    f"success rate={metadata.get('bootstrap_success_rate', 'unknown')}, "
                    f"leave-one-group-out={metadata.get('successful_leave_one_group_out', 'unknown')}"
                )
                warnings.extend(getattr(report, "warnings", []))
            score = 12 if basic_exists and advanced_exists else 8
            if warnings:
                score = max(score - 3, 5)
            return AuditItem(
                category="모형 검증",
                item="강건성 분석",
                status="PASS" if not warnings else "WARNING",
                score=score,
                maximum_score=12,
                evidence="; ".join(methods),
                recommendation="Report optimizer sensitivity, group bootstrap, and leave-one-group-out stability.",
            )
        return AuditItem(
            category="모형 검증",
            item="강건성 분석",
            status="NOT_APPLICABLE",
            score=0,
            maximum_score=0,
            evidence=("현재 자동 강건성 분석은 혼합효과모형에 적용되지 않습니다."),
            recommendation=(
                "필요한 경우 ML/REML 비교, 대체 공분산 구조 또는 "
                "그룹 제외 민감도 분석을 별도로 수행하세요."
            ),
        )

    basic = _artifact_exists(
        runtime,
        f"robustness_report:{model_id}",
    )
    bootstrap = _artifact_exists(
        runtime,
        f"bootstrap_report:{model_id}",
    )
    jackknife = _artifact_exists(
        runtime,
        f"jackknife_report:{model_id}",
    )
    cluster = _artifact_exists(
        runtime,
        f"cluster_report:{model_id}",
    )

    score = 0
    methods: list[str] = []

    if basic:
        score += 4
        methods.append("HC 비교")
    if bootstrap:
        score += 4
        methods.append("Bootstrap")
    if jackknife:
        score += 2
        methods.append("Jackknife")
    if cluster:
        score += 2
        methods.append("Cluster robust")

    score = min(score, 12)

    if score >= 10:
        status = "PASS"
    elif score >= 4:
        status = "WARNING"
    else:
        status = "MISSING"

    return AuditItem(
        category="모형 검증",
        item="강건성 분석",
        status=status,
        score=score,
        maximum_score=12,
        evidence=(", ".join(methods) if methods else "강건성 분석 결과가 없습니다."),
        recommendation=("HC3, Bootstrap, Cluster 구조에 맞는 강건성 검정을 보고하세요."),
    )


def _effect_size_item(
    runtime: PipelineRuntime,
    model_id: str,
) -> AuditItem:
    key = f"effect_size_report:{model_id}"

    if not _artifact_exists(runtime, key):
        return AuditItem(
            category="결과 보고",
            item="효과크기",
            status="MISSING",
            score=0,
            maximum_score=10,
            evidence="효과크기 결과가 없습니다.",
            recommendation="표준화 β, 부분 R², OR 또는 AME를 보고하세요.",
        )

    report = runtime.artifacts[key]





    if getattr(report, "model_type", None) == "exponential_aft":
        model_effects = getattr(report, "model_effects", {})
        evidence = (
            f"Exponential AFT time-ratio effects {len(report.effects)} generated; "
            f"constant hazard={model_effects.get('constant_hazard', 'unknown')}"
        )
        recommendation = "Interpret time ratios under the constant-hazard exponential AFT model."
    elif getattr(report, "model_type", None) == "loglogistic_aft":
        model_effects = getattr(report, "model_effects", {})
        evidence = (
            f"Log-logistic AFT time-ratio effects {len(report.effects)} generated; "
            f"shape={model_effects.get('shape', 'unknown')}"
        )
        recommendation = "Interpret time ratios as acceleration or deceleration of median survival time."
    elif getattr(report, "model_type", None) == "lognormal_aft":
        model_effects = getattr(report, "model_effects", {})
        evidence = (
            f"Log-normal AFT time-ratio effects {len(report.effects)} generated; "
            f"sigma={model_effects.get('sigma', 'unknown')}"
        )
        recommendation = "Interpret time ratios as acceleration or deceleration of median survival time."
    elif getattr(report, "model_type", None) == "weibull_ph":
        model_effects = getattr(report, "model_effects", {})
        evidence = (
            f"Weibull PH hazard-ratio effects {len(report.effects)} generated; "
            f"shape={model_effects.get('shape', 'unknown')}"
        )
        recommendation = "Interpret hazard ratios under the Weibull proportional hazards parameterization."
    elif getattr(report, "model_type", None) == "weibull_aft":
        model_effects = getattr(report, "model_effects", {})
        evidence = (
            f"Weibull AFT time-ratio effects {len(report.effects)} generated; "
            f"shape={model_effects.get('shape', 'unknown')}"
        )
        recommendation = "Interpret time ratios as acceleration or deceleration of survival time."
    elif getattr(report, "model_type", None) == "stratified_cox":
        model_effects = getattr(report, "model_effects", {})
        metadata = getattr(report, "metadata", {})
        evidence = (
            f"Stratified Cox hazard-ratio effects {len(report.effects)} generated; "
            f"strata={metadata.get('strata_count', 'unknown')}"
        )
        recommendation = "Interpret hazard ratios conditional on strata-specific baseline hazards."
    elif getattr(report, "model_type", None) == "left_truncated_cox":
        model_effects = getattr(report, "model_effects", {})
        metadata = getattr(report, "metadata", {})
        evidence = (
            f"Left-truncated Cox hazard-ratio effects {len(report.effects)} generated; "
            f"entry variable={metadata.get('entry_variable', 'unknown')}"
        )
        recommendation = "Interpret hazard ratios with delayed-entry risk sets."
    elif getattr(report, "model_type", None) == "cause_specific_cox":
        model_effects = getattr(report, "model_effects", {})
        metadata = getattr(report, "metadata", {})
        evidence = (
            f"Cause-specific Cox hazard-ratio effects {len(report.effects)} generated; "
            f"target={metadata.get('target_event_code', 'unknown')}"
        )
        recommendation = "Interpret hazard ratios for the target event while competing events are censored."
    elif getattr(report, "model_type", None) == "clustered_cox":
        model_effects = getattr(report, "model_effects", {})
        metadata = getattr(report, "metadata", {})
        evidence = (
            f"Clustered Cox hazard-ratio effects {len(report.effects)} generated; "
            f"clusters={metadata.get('cluster_count', 'unknown')}"
        )
        recommendation = "Interpret hazard ratios with cluster-robust standard errors."
    elif getattr(report, "model_type", None) == "time_varying_cox":
        model_effects = getattr(report, "model_effects", {})
        metadata = getattr(report, "metadata", {})
        evidence = (
            f"Time-varying Cox hazard-ratio effects {len(report.effects)} generated; "
            f"rows={metadata.get('time_varying_row_count', 'unknown')}"
        )
        recommendation = "Interpret hazard ratios from start-stop risk intervals and report subject clustering when used."
    elif getattr(report, "model_type", None) == "piecewise_exponential":
        model_effects = getattr(report, "model_effects", {})
        evidence = (
            f"Piecewise exponential hazard-ratio effects {len(report.effects)} generated; "
            f"intervals={model_effects.get('interval_count', 'unknown')}"
        )
        recommendation = "Interpret hazard ratios with interval-specific baseline hazards and log-exposure offsets."
    elif getattr(report, "model_type", None) == "discrete_time_hazard":
        model_effects = getattr(report, "model_effects", {})
        metadata = getattr(report, "metadata", {})
        evidence = (
            f"Discrete-time hazard effects {len(report.effects)} generated; "
            f"intervals={model_effects.get('interval_count', 'unknown')}, link={metadata.get('link', 'unknown')}"
        )
        recommendation = "Interpret effects on interval-specific discrete hazards from person-period data."
    elif getattr(report, "model_type", None) == "log_binomial":
        model_effects = getattr(report, "model_effects", {})
        evidence = (
            f"Log-binomial risk-ratio effects {len(report.effects)} generated; "
            f"Brier={model_effects.get('brier_score', 'unknown')}"
        )
        recommendation = "Interpret risk ratios and average marginal effects on event probability."
    elif getattr(report, "model_type", None) == "weighted_least_squares":
        model_effects = getattr(report, "model_effects", {})
        evidence = (
            f"WLS standardized effects {len(report.effects)} generated; "
            f"R-squared={model_effects.get('r_squared', 'unknown')}"
        )
        recommendation = "Interpret standardized WLS coefficients with the analytic weights and residual diagnostics."
    elif getattr(report, "model_type", None) == "binary_cloglog":
        model_effects = getattr(report, "model_effects", {})
        evidence = (
            f"Binary cloglog effects {len(report.effects)} generated; "
            f"Brier={model_effects.get('brier_score', 'unknown')}"
        )
        recommendation = "Interpret exponentiated cloglog coefficients and average marginal effects on event probability."
    elif getattr(report, "model_type", None) == "ordered_probit":
        model_effects = getattr(report, "model_effects", {})
        evidence = (
            f"Ordered probit latent effects {len(report.effects)} generated; "
            f"categories={model_effects.get('category_count', 'unknown')}"
        )
        recommendation = "Interpret latent-index coefficients with ordered thresholds and category prediction diagnostics."
    elif getattr(report, "model_type", None) == "binary_probit":
        model_effects = getattr(report, "model_effects", {})
        evidence = (
            f"Binary probit effects {len(report.effects)} generated; "
            f"Brier={model_effects.get('brier_score', 'unknown')}"
        )
        recommendation = "Interpret average marginal effects on event probability alongside latent-index coefficients."
    elif getattr(report, "model_type", None) == "heckman_selection":
        model_effects = getattr(report, "model_effects", {})
        evidence = (
            f"Heckman standardized effects {len(report.effects)} generated; "
            f"inverse Mills p={model_effects.get('inverse_mills_p_value', 'unknown')}"
        )
        recommendation = "Interpret outcome-equation standardized coefficients with the selection correction."
    elif getattr(report, "model_type", None) == "iv_2sls_regression":
        model_effects = getattr(report, "model_effects", {})
        evidence = (
            f"IV standardized effects {len(report.effects)} generated; "
            f"minimum first-stage F={model_effects.get('minimum_first_stage_f_statistic', 'unknown')}"
        )
        recommendation = "Interpret IV coefficients with instrument strength and exclusion restrictions."
    elif getattr(report, "model_type", None) == "inverse_gaussian_regression":
        model_effects = getattr(report, "model_effects", {})
        evidence = (
            f"Inverse Gaussian mean-ratio effects {len(report.effects)} generated; "
            f"dispersion={model_effects.get('dispersion_ratio', 'unknown')}"
        )
        recommendation = "Interpret multiplicative mean ratios from the Inverse Gaussian log-link model."
    elif getattr(report, "model_type", None) == "gamma_regression":
        model_effects = getattr(report, "model_effects", {})
        evidence = (
            f"Gamma mean-ratio effects {len(report.effects)} generated; "
            f"dispersion={model_effects.get('dispersion_ratio', 'unknown')}"
        )
        recommendation = "Interpret multiplicative mean ratios from the Gamma log-link model."
    elif getattr(report, "model_type", None) == "regularized_regression":
        model_effects = getattr(report, "model_effects", {})
        evidence = (
            f"Regularized standardized effects {len(report.effects)} generated; "
            f"selected={model_effects.get('selected_coefficient_count', 'unknown')}"
        )
        recommendation = "Interpret penalized standardized coefficients with tuning and selection diagnostics."
    elif getattr(report, "model_type", None) == "robust_regression":
        model_effects = getattr(report, "model_effects", {})
        evidence = (
            f"Robust standardized effects {len(report.effects)} generated; "
            f"downweighted={model_effects.get('downweighted_count', 'unknown')}"
        )
        recommendation = "Interpret robust standardized coefficients with the weight diagnostics."
    elif getattr(report, "model_type", None) == "tobit_regression":
        model_effects = getattr(report, "model_effects", {})
        evidence = (
            f"Tobit effect sizes {len(report.effects)} generated; "
            f"censoring rate={model_effects.get('censoring_rate', 'unknown')}"
        )
        recommendation = "Interpret latent standardized coefficients and observed-scale marginal effects."
    elif getattr(report, "model_type", None) in {"panel_fixed_effects", "panel_random_effects", "panel_correlated_random_effects", "panel_between_effects", "panel_first_difference", "panel_pooled_ols"}:
        model_effects = getattr(report, "model_effects", {})
        panel_r_squared = model_effects.get(
            "within_r_squared",
            model_effects.get("marginal_r_squared", model_effects.get("between_r_squared", "unknown")),
        )
        evidence = (
            f"Panel effect sizes {len(report.effects)} generated; "
            f"panel R-squared={panel_r_squared}"
        )
        recommendation = "Interpret standardized panel coefficients for the requested panel estimand."
    elif getattr(report, "model_type", None) in {
        "mixed_random_intercept",
        "mixed_random_slope",
        "mixed_three_level",
    }:
        model_effects = getattr(report, "model_effects", {})
        available = [
            name
            for name in (
                "intraclass_correlation",
                "marginal_r_squared",
                "conditional_r_squared",
            )
            if name in model_effects
        ]
        evidence = (
            f"표준화 고정효과 {len(report.effects)}건 및 모형 효과크기 {', '.join(available)} 생성"
        )
        recommendation = "표준화 고정효과와 함께 ICC, marginal R², conditional R²를 해석하세요."
    else:
        evidence = f"효과크기 {len(report.effects)}건 생성"
        recommendation = "유의확률과 함께 효과크기의 실질적 의미를 해석하세요."

    return AuditItem(
        category="결과 보고",
        item="효과크기",
        status="PASS",
        score=10,
        maximum_score=10,
        evidence=evidence,
        recommendation=recommendation,
    )


def _reporting_item(
    runtime: PipelineRuntime,
    model_id: str,
) -> AuditItem:
    key = f"regression_publication_report:{model_id}"

    if not _artifact_exists(runtime, key):
        return AuditItem(
            category="결과 보고",
            item="논문용 표·서술",
            status="MISSING",
            score=0,
            maximum_score=8,
            evidence="논문용 보고서 결과가 없습니다.",
            recommendation="논문용 표와 결과 서술 초안을 생성하세요.",
        )

    return AuditItem(
        category="결과 보고",
        item="논문용 표·서술",
        status="PASS",
        score=8,
        maximum_score=8,
        evidence="논문용 표와 한국어 결과문이 생성되었습니다.",
        recommendation="학술지 양식에 맞게 표 번호와 주석을 최종 조정하세요.",
    )


def _visualization_item(
    runtime: PipelineRuntime,
    model_id: str,
) -> AuditItem:
    key = f"regression_visualization:{model_id}"

    if not _artifact_exists(runtime, key):
        return AuditItem(
            category="결과 보고",
            item="시각화",
            status="REVIEW",
            score=2,
            maximum_score=5,
            evidence="시각화 결과가 없습니다.",
            recommendation="필요한 경우 계수·잔차·영향력 도표를 추가하세요.",
        )

    report = runtime.artifacts[key]

    return AuditItem(
        category="결과 보고",
        item="시각화",
        status="PASS",
        score=5,
        maximum_score=5,
        evidence=f"그림 {report.metadata.get('figure_count', 0)}개 생성",
        recommendation="본문 또는 부록에 필요한 그림만 선별하여 사용하세요.",
    )


def build_research_audit_report(
    runtime: PipelineRuntime,
    *,
    model_id: str = "main_model",
) -> ResearchAuditReport:
    """현재 Runtime 산출물의 연구 품질 감사를 수행한다."""
    items = [
        _missingness_item(runtime),
        _outlier_item(runtime),
        _reliability_item(runtime),
        _regression_item(runtime, model_id),
        _diagnostics_item(runtime, model_id),
    ]
    panel_hausman_item = _panel_hausman_item(runtime, model_id)
    if panel_hausman_item is not None:
        items.append(panel_hausman_item)
    items.extend(
        [
            _robustness_item(runtime, model_id),
            _effect_size_item(runtime, model_id),
            _reporting_item(runtime, model_id),
            _visualization_item(runtime, model_id),
        ]
    )

    total_score = sum(item.score for item in items)
    maximum_score = sum(item.maximum_score for item in items)
    percentage = total_score / maximum_score * 100 if maximum_score else 0.0
    warnings = [
        f"{item.category} - {item.item}: {item.recommendation}"
        for item in items
        if item.status
        in {
            "MISSING",
            "FAIL",
            "WARNING",
        }
    ]

    regression_result = _regression_result(runtime, model_id)
    metadata: dict[str, Any] = {
        "audit_item_count": len(items),
        "passed_item_count": sum(item.status == "PASS" for item in items),
        "not_applicable_item_count": sum(item.status == "NOT_APPLICABLE" for item in items),
    }
    panel_hausman = runtime.artifacts.get(f"panel_hausman:{model_id}")
    if panel_hausman is not None:
        metadata.update(
            {
                "panel_hausman_statistic": getattr(panel_hausman, "statistic", None),
                "panel_hausman_degrees_of_freedom": getattr(panel_hausman, "degrees_of_freedom", None),
                "panel_hausman_p_value": getattr(panel_hausman, "p_value", None),
                "panel_hausman_status": getattr(panel_hausman, "status", None),
            }
        )

    if regression_result is not None:
        metadata["model_type"] = regression_result.model_type
        metadata["selected_survival_model"] = regression_result.metadata.get("selected_survival_model")
        metadata["survival_selection_criterion"] = regression_result.metadata.get("survival_selection_criterion")
        metadata["candidate_survival_model_count"] = regression_result.metadata.get("candidate_survival_model_count")
        metadata["candidate_survival_models"] = regression_result.metadata.get("candidate_survival_models")
        if regression_result.model_type in _THREE_LEVEL_MIXED_MODELS:
            metadata.update(
                {
                    "level2_group": regression_result.metadata.get("level2_group"),
                    "level3_group": regression_result.metadata.get("level3_group"),
                    "level2_group_count": regression_result.fit_statistics.get(
                        "level2_group_count"
                    ),
                    "level3_group_count": regression_result.fit_statistics.get(
                        "level3_group_count"
                    ),
                    "level2_vpc": regression_result.fit_statistics.get(
                        "level2_vpc",
                        regression_result.fit_statistics.get("level2_intraclass_correlation"),
                    ),
                    "level3_vpc": regression_result.fit_statistics.get(
                        "level3_vpc",
                        regression_result.fit_statistics.get("level3_intraclass_correlation"),
                    ),
                }
            )
        elif regression_result.model_type == "beta_regression":
            metadata.update(
                {
                    "precision": regression_result.fit_statistics.get("precision"),
                    "pseudo_r_squared": regression_result.fit_statistics.get("pseudo_r_squared"),
                    "mean_absolute_error": regression_result.fit_statistics.get("mean_absolute_error"),
                }
            )
        elif regression_result.model_type == "fractional_logit":
            metadata.update(
                {
                    "boundary_count": regression_result.fit_statistics.get("boundary_count"),
                    "dispersion_ratio": regression_result.fit_statistics.get("dispersion_ratio"),
                    "pseudo_r_squared_deviance": regression_result.fit_statistics.get("pseudo_r_squared_deviance"),
                }
            )
        elif regression_result.model_type in {"cox_proportional_hazards", "stratified_cox", "left_truncated_cox", "cause_specific_cox", "clustered_cox", "time_varying_cox", "piecewise_exponential", "discrete_time_hazard"}:
            metadata.update(
                {
                    "duration_variable": regression_result.metadata.get("duration_variable"),
                    "event_variable": regression_result.metadata.get("event_variable"),
                    "start_variable": regression_result.metadata.get("start_variable"),
                    "stop_variable": regression_result.metadata.get("stop_variable"),
                    "subject_variable": regression_result.metadata.get("subject_variable"),
                    "subject_count": regression_result.fit_statistics.get("subject_count"),
                    "time_varying_row_count": regression_result.fit_statistics.get("time_varying_row_count"),
                    "interval_count": regression_result.fit_statistics.get("interval_count"),
                    "interval_breakpoints": regression_result.metadata.get("interval_breakpoints"),
                    "discrete_time_link": regression_result.metadata.get("link") if regression_result.model_type == "discrete_time_hazard" else None,
                    "person_period_event_rate": regression_result.fit_statistics.get("person_period_event_rate"),
                    "total_exposure": regression_result.fit_statistics.get("total_exposure"),
                    "cause_variable": regression_result.metadata.get("cause_variable"),
                    "target_event_code": regression_result.metadata.get("target_event_code"),
                    "competing_event_count": regression_result.fit_statistics.get("competing_event_count"),
                    "cluster_variable": regression_result.metadata.get("cluster_variable"),
                    "cluster_count": regression_result.fit_statistics.get("cluster_count"),
                    "event_count": regression_result.fit_statistics.get("event_count"),
                    "censored_count": regression_result.fit_statistics.get("censored_count"),
                    "events_per_parameter": regression_result.fit_statistics.get("events_per_parameter"),
                    "entry_variable": regression_result.metadata.get("entry_variable"),
                    "left_truncated_count": regression_result.fit_statistics.get("left_truncated_count"),
                    "strata_variable": regression_result.metadata.get("strata_variable"),
                    "strata_count": regression_result.fit_statistics.get("strata_count", regression_result.metadata.get("strata_count")),
                }
            )




        elif regression_result.model_type == "exponential_aft":
            metadata.update(
                {
                    "duration_variable": regression_result.metadata.get("duration_variable"),
                    "event_variable": regression_result.metadata.get("event_variable"),
                    "event_count": regression_result.fit_statistics.get("event_count"),
                    "censored_count": regression_result.fit_statistics.get("censored_count"),
                    "events_per_parameter": regression_result.fit_statistics.get("events_per_parameter"),
                    "strata_variable": regression_result.metadata.get("strata_variable"),
                    "strata_count": regression_result.fit_statistics.get("strata_count", regression_result.metadata.get("strata_count")),
                    "constant_hazard": regression_result.fit_statistics.get("constant_hazard"),
                    "median_predicted_time": regression_result.fit_statistics.get("median_predicted_time"),
                }
            )
        elif regression_result.model_type == "loglogistic_aft":
            metadata.update(
                {
                    "duration_variable": regression_result.metadata.get("duration_variable"),
                    "event_variable": regression_result.metadata.get("event_variable"),
                    "event_count": regression_result.fit_statistics.get("event_count"),
                    "censored_count": regression_result.fit_statistics.get("censored_count"),
                    "events_per_parameter": regression_result.fit_statistics.get("events_per_parameter"),
                    "strata_variable": regression_result.metadata.get("strata_variable"),
                    "strata_count": regression_result.fit_statistics.get("strata_count", regression_result.metadata.get("strata_count")),
                    "loglogistic_shape": regression_result.fit_statistics.get("shape"),
                    "median_predicted_time": regression_result.fit_statistics.get("median_predicted_time"),
                }
            )
        elif regression_result.model_type == "lognormal_aft":
            metadata.update(
                {
                    "duration_variable": regression_result.metadata.get("duration_variable"),
                    "event_variable": regression_result.metadata.get("event_variable"),
                    "event_count": regression_result.fit_statistics.get("event_count"),
                    "censored_count": regression_result.fit_statistics.get("censored_count"),
                    "events_per_parameter": regression_result.fit_statistics.get("events_per_parameter"),
                    "strata_variable": regression_result.metadata.get("strata_variable"),
                    "strata_count": regression_result.fit_statistics.get("strata_count", regression_result.metadata.get("strata_count")),
                    "lognormal_sigma": regression_result.fit_statistics.get("sigma"),
                    "median_predicted_time": regression_result.fit_statistics.get("median_predicted_time"),
                }
            )
        elif regression_result.model_type in {"weibull_aft", "weibull_ph"}:
            metadata.update(
                {
                    "duration_variable": regression_result.metadata.get("duration_variable"),
                    "event_variable": regression_result.metadata.get("event_variable"),
                    "event_count": regression_result.fit_statistics.get("event_count"),
                    "censored_count": regression_result.fit_statistics.get("censored_count"),
                    "events_per_parameter": regression_result.fit_statistics.get("events_per_parameter"),
                    "strata_variable": regression_result.metadata.get("strata_variable"),
                    "strata_count": regression_result.fit_statistics.get("strata_count", regression_result.metadata.get("strata_count")),
                    "weibull_shape": regression_result.fit_statistics.get("shape"),
                    "weibull_baseline_rate": regression_result.fit_statistics.get("baseline_rate"),
                    "median_predicted_time": regression_result.fit_statistics.get("median_predicted_time"),
                }
            )
        elif regression_result.model_type == "quantile_regression":
            metadata.update(
                {
                    "quantile": regression_result.fit_statistics.get("quantile"),
                    "pseudo_r_squared": regression_result.fit_statistics.get("pseudo_r_squared"),
                    "pinball_loss": regression_result.fit_statistics.get("pinball_loss"),
                }
            )
        elif regression_result.model_type == "tweedie_regression":
            metadata.update(
                {
                    "variance_power": regression_result.metadata.get("variance_power"),
                    "zero_count": regression_result.fit_statistics.get("zero_count"),
                    "zero_proportion": regression_result.fit_statistics.get("zero_proportion"),
                    "dispersion_ratio": regression_result.fit_statistics.get("dispersion_ratio"),
                    "pseudo_r_squared_deviance": regression_result.fit_statistics.get("pseudo_r_squared_deviance"),
                }
            )
        elif regression_result.model_type == "multinomial_logit":
            metadata.update(
                {
                    "category_count": regression_result.fit_statistics.get("category_count"),
                    "reference_category": regression_result.metadata.get("reference_category"),
                    "category_labels": regression_result.metadata.get("category_labels"),
                }
            )
        elif regression_result.model_type in {
            "poisson",
            "quasi_poisson",
            "negative_binomial",
            "generalized_poisson",
            "zero_inflated_poisson",
            "zero_inflated_negative_binomial",
            "hurdle_poisson",
            "hurdle_negative_binomial",
        }:
            metadata.update(
                {
                    "zero_count": regression_result.fit_statistics.get("zero_count"),
                    "zero_proportion": regression_result.fit_statistics.get("zero_proportion"),
                    "predicted_zero_proportion": regression_result.fit_statistics.get("predicted_zero_proportion"),
                    "alpha": regression_result.fit_statistics.get("alpha"),
                    "selected_count_model": regression_result.metadata.get("selected_count_model"),
                    "inflation_model": regression_result.metadata.get("inflation_model"),
                    "hurdle_model": regression_result.metadata.get("hurdle_model"),
                    "count_model": regression_result.metadata.get("count_model"),
                    "generalized_poisson_parameterization": regression_result.metadata.get("generalized_poisson_parameterization"),
                    "quasi_likelihood": regression_result.metadata.get("quasi_likelihood"),
                    "scale": regression_result.fit_statistics.get("scale"),
                }
            )
        elif regression_result.model_type in {"gee_gaussian", "gee_logit", "gee_poisson", "gee_negative_binomial", "gee_gamma", "gee_inverse_gaussian", "gee_tweedie"}:
            metadata.update(
                {
                    "group_variable": regression_result.metadata.get("group_variable"),
                    "cluster_count": regression_result.fit_statistics.get("cluster_count"),
                    "covariance_structure": regression_result.metadata.get("covariance_structure"),
                    "negative_binomial_alpha": regression_result.fit_statistics.get("negative_binomial_alpha"),
                    "gamma_scale": regression_result.fit_statistics.get("gamma_scale"),
                    "inverse_gaussian_scale": regression_result.fit_statistics.get("inverse_gaussian_scale"),
                    "tweedie_var_power": regression_result.fit_statistics.get("tweedie_var_power"),
                    "tweedie_scale": regression_result.fit_statistics.get("tweedie_scale"),
                }
            )
        elif regression_result.model_type in {"log_binomial", "quasi_binomial", "modified_poisson", "linear_probability_model"}:
            metadata.update(
                {
                    "link": regression_result.metadata.get("link"),
                    "event_count": regression_result.fit_statistics.get("event_count"),
                    "non_event_count": regression_result.fit_statistics.get("non_event_count"),
                    "pseudo_r_squared_mcfadden": regression_result.fit_statistics.get("pseudo_r_squared_mcfadden"),
                    "brier_score": regression_result.fit_statistics.get("brier_score"),
                    "out_of_bounds_prediction_count": regression_result.fit_statistics.get("out_of_bounds_prediction_count"),
                    "modified_poisson": regression_result.metadata.get("modified_poisson"),
                    "quasi_binomial": regression_result.model_type == "quasi_binomial",
                    "dispersion_scale": regression_result.fit_statistics.get("dispersion_scale"),
                    "linear_probability_model": regression_result.model_type == "linear_probability_model",
                    "r_squared": regression_result.fit_statistics.get("r_squared"),
                }
            )
        elif regression_result.model_type == "weighted_least_squares":
            metadata.update(
                {
                    "weight_variable": regression_result.metadata.get("weight_variable"),
                    "weight_sum": regression_result.fit_statistics.get("weight_sum"),
                    "weight_ratio": regression_result.fit_statistics.get("weight_ratio"),
                    "r_squared": regression_result.fit_statistics.get("r_squared"),
                }
            )
        elif regression_result.model_type == "binary_cloglog":
            metadata.update(
                {
                    "link": regression_result.metadata.get("link"),
                    "event_count": regression_result.fit_statistics.get("event_count"),
                    "non_event_count": regression_result.fit_statistics.get("non_event_count"),
                    "pseudo_r_squared_mcfadden": regression_result.fit_statistics.get("pseudo_r_squared_mcfadden"),
                    "brier_score": regression_result.fit_statistics.get("brier_score"),
                }
            )
        elif regression_result.model_type == "ordered_probit":
            metadata.update(
                {
                    "link": regression_result.metadata.get("link"),
                    "category_count": regression_result.fit_statistics.get("category_count"),
                    "category_counts": regression_result.metadata.get("category_counts"),
                    "threshold_terms": regression_result.metadata.get("threshold_terms"),
                }
            )
        elif regression_result.model_type == "binary_probit":
            metadata.update(
                {
                    "link": regression_result.metadata.get("link"),
                    "event_count": regression_result.fit_statistics.get("event_count"),
                    "non_event_count": regression_result.fit_statistics.get("non_event_count"),
                    "pseudo_r_squared_mcfadden": regression_result.fit_statistics.get("pseudo_r_squared_mcfadden"),
                    "brier_score": regression_result.fit_statistics.get("brier_score"),
                }
            )
        elif regression_result.model_type == "heckman_selection":
            metadata.update(
                {
                    "selection_variable": regression_result.metadata.get("selection_variable"),
                    "selection_variables": regression_result.metadata.get("selection_variables"),
                    "exclusion_restrictions": regression_result.metadata.get("exclusion_restrictions"),
                    "selection_rate": regression_result.fit_statistics.get("selection_rate"),
                    "inverse_mills_coefficient": regression_result.fit_statistics.get("inverse_mills_coefficient"),
                    "inverse_mills_p_value": regression_result.fit_statistics.get("inverse_mills_p_value"),
                    "rho": regression_result.fit_statistics.get("rho"),
                }
            )
        elif regression_result.model_type == "iv_2sls_regression":
            metadata.update(
                {
                    "endogenous_variables": regression_result.metadata.get("endogenous_variables"),
                    "instrument_variables": regression_result.metadata.get("instrument_variables"),
                    "instrument_count": regression_result.fit_statistics.get("instrument_count"),
                    "minimum_first_stage_f_statistic": regression_result.fit_statistics.get("minimum_first_stage_f_statistic"),
                    "weak_instrument_warning": bool(
                        regression_result.fit_statistics.get("minimum_first_stage_f_statistic") is not None
                        and float(regression_result.fit_statistics.get("minimum_first_stage_f_statistic")) < 10.0
                    ),
                }
            )
        elif regression_result.model_type == "inverse_gaussian_regression":
            metadata.update(
                {
                    "dispersion_ratio": regression_result.fit_statistics.get("dispersion_ratio"),
                    "pseudo_r_squared_deviance": regression_result.fit_statistics.get("pseudo_r_squared_deviance"),
                    "root_mean_squared_error": regression_result.fit_statistics.get("root_mean_squared_error"),
                    "minimum_observed": regression_result.fit_statistics.get("minimum_observed"),
                    "maximum_observed": regression_result.fit_statistics.get("maximum_observed"),
                }
            )
        elif regression_result.model_type == "gamma_regression":
            metadata.update(
                {
                    "dispersion_ratio": regression_result.fit_statistics.get("dispersion_ratio"),
                    "pseudo_r_squared_deviance": regression_result.fit_statistics.get("pseudo_r_squared_deviance"),
                    "root_mean_squared_error": regression_result.fit_statistics.get("root_mean_squared_error"),
                    "minimum_observed": regression_result.fit_statistics.get("minimum_observed"),
                    "maximum_observed": regression_result.fit_statistics.get("maximum_observed"),
                }
            )
        elif regression_result.model_type == "regularized_regression":
            metadata.update(
                {
                    "penalty": regression_result.fit_statistics.get("penalty"),
                    "alpha": regression_result.fit_statistics.get("alpha"),
                    "l1_ratio": regression_result.fit_statistics.get("l1_ratio"),
                    "selected_coefficient_count": regression_result.fit_statistics.get("selected_coefficient_count"),
                    "zero_coefficient_count": regression_result.fit_statistics.get("zero_coefficient_count"),
                    "root_mean_squared_error": regression_result.fit_statistics.get("root_mean_squared_error"),
                }
            )
        elif regression_result.model_type == "robust_regression":
            metadata.update(
                {
                    "robust_norm": regression_result.metadata.get("norm"),
                    "downweighted_count": regression_result.fit_statistics.get("downweighted_count"),
                    "heavily_downweighted_count": regression_result.fit_statistics.get("heavily_downweighted_count"),
                    "downweighted_rate": regression_result.fit_statistics.get("downweighted_rate"),
                    "pseudo_r_squared": regression_result.fit_statistics.get("pseudo_r_squared"),
                }
            )
        elif regression_result.model_type == "tobit_regression":
            metadata.update(
                {
                    "lower_limit": regression_result.metadata.get("lower_limit"),
                    "upper_limit": regression_result.metadata.get("upper_limit"),
                    "left_censored_count": regression_result.fit_statistics.get("left_censored_count"),
                    "right_censored_count": regression_result.fit_statistics.get("right_censored_count"),
                    "censoring_rate": regression_result.fit_statistics.get("censoring_rate"),
                    "sigma": regression_result.fit_statistics.get("sigma"),
                }
            )
        elif regression_result.model_type in {"panel_fixed_effects", "panel_random_effects", "panel_correlated_random_effects", "panel_between_effects", "panel_first_difference", "panel_pooled_ols"}:
            metadata.update(
                {
                    "entity_variable": regression_result.metadata.get("entity_variable"),
                    "time_variable": regression_result.metadata.get("time_variable"),
                    "entity_count": regression_result.fit_statistics.get("entity_count"),
                    "time_period_count": regression_result.fit_statistics.get("time_period_count"),
                    "within_r_squared": regression_result.fit_statistics.get("within_r_squared"),
                    "marginal_r_squared": regression_result.fit_statistics.get("marginal_r_squared"),
                    "between_r_squared": regression_result.fit_statistics.get("between_r_squared"),
                    "first_difference_r_squared": regression_result.fit_statistics.get("first_difference_r_squared"),
                    "pooled_r_squared": regression_result.fit_statistics.get("pooled_r_squared"),
                    "adjusted_pooled_r_squared": regression_result.fit_statistics.get("adjusted_pooled_r_squared"),
                    "adjusted_first_difference_r_squared": regression_result.fit_statistics.get("adjusted_first_difference_r_squared"),
                    "differenced_entity_count": regression_result.fit_statistics.get("differenced_entity_count"),
                    "adjusted_between_r_squared": regression_result.fit_statistics.get("adjusted_between_r_squared"),
                    "conditional_r_squared": regression_result.fit_statistics.get("conditional_r_squared"),
                    "random_intercept_variance": regression_result.fit_statistics.get("random_intercept_variance"),
                    "entity_mean_term_count": regression_result.fit_statistics.get("entity_mean_term_count"),
                }
            )
        elif _is_mixed_effects_model(runtime, model_id):
            metadata.update(
                {
                    "group_variable": regression_result.metadata.get("group_variable"),
                    "group_count": regression_result.fit_statistics.get("group_count"),
                    "intraclass_correlation": regression_result.fit_statistics.get(
                        "intraclass_correlation",
                        regression_result.fit_statistics.get("icc"),
                    ),
                }
            )

    return ResearchAuditReport(
        model_id=model_id,
        items=items,
        total_score=total_score,
        maximum_score=maximum_score,
        percentage=percentage,
        grade=_grade_from_percentage(percentage),
        submission_status=_submission_status(percentage),
        warnings=warnings,
        metadata=metadata,
    )


def audit_items_to_dataframe(
    report: ResearchAuditReport,
) -> pd.DataFrame:
    """감사 항목을 데이터프레임으로 변환한다."""
    return pd.DataFrame([asdict(item) for item in report.items])


def audit_summary_to_dataframe(
    report: ResearchAuditReport,
) -> pd.DataFrame:
    """감사 요약을 세로형 표로 변환한다."""
    values = {
        "model_id": report.model_id,
        "total_score": report.total_score,
        "maximum_score": report.maximum_score,
        "percentage": report.percentage,
        "grade": report.grade,
        "submission_status": report.submission_status,
        "warning_count": len(report.warnings),
        **report.metadata,
    }

    return pd.DataFrame(
        {
            "item": list(values.keys()),
            "value": list(values.values()),
        }
    )


def write_audit_narrative(
    report: ResearchAuditReport,
) -> str:
    """한국어 연구 감사 요약문을 생성한다."""
    lines = [
        "연구 품질 감사 결과",
        "",
        (
            f"총점은 {report.total_score}/{report.maximum_score}점"
            f"({report.percentage:.1f}%)이며, 등급은 {report.grade}이다."
        ),
        f"종합 판정은 '{report.submission_status}'이다.",
        "",
        "보완 권고:",
    ]

    if report.warnings:
        lines.extend(f"- {warning}" for warning in report.warnings)
    else:
        lines.append("- 주요 보완 권고 없음")

    return "\n".join(lines)
