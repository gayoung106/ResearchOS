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
    if result is not None and getattr(result, "model_type", None) in {"gee_gaussian", "gee_logit", "gee_poisson"}:
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

    if getattr(report, "model_type", None) in {
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
        _robustness_item(runtime, model_id),
        _effect_size_item(runtime, model_id),
        _reporting_item(runtime, model_id),
        _visualization_item(runtime, model_id),
    ]

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
    if regression_result is not None:
        metadata["model_type"] = regression_result.model_type
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
        elif regression_result.model_type == "cox_proportional_hazards":
            metadata.update(
                {
                    "duration_variable": regression_result.metadata.get("duration_variable"),
                    "event_variable": regression_result.metadata.get("event_variable"),
                    "event_count": regression_result.fit_statistics.get("event_count"),
                    "censored_count": regression_result.fit_statistics.get("censored_count"),
                    "events_per_parameter": regression_result.fit_statistics.get("events_per_parameter"),
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
        elif regression_result.model_type == "multinomial_logit":
            metadata.update(
                {
                    "category_count": regression_result.fit_statistics.get("category_count"),
                    "reference_category": regression_result.metadata.get("reference_category"),
                    "category_labels": regression_result.metadata.get("category_labels"),
                }
            )
        elif regression_result.model_type in {"gee_gaussian", "gee_logit", "gee_poisson"}:
            metadata.update(
                {
                    "group_variable": regression_result.metadata.get("group_variable"),
                    "cluster_count": regression_result.fit_statistics.get("cluster_count"),
                    "covariance_structure": regression_result.metadata.get("covariance_structure"),
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
