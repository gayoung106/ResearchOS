"""테스트에서 사용하는 예상 파이프라인 구성."""

from __future__ import annotations

DATA_LOADING = "01_data_loading"
VARIABLE_DETECTION = "02_variable_detection"
EVIDENCE_RESOLUTION = "02_evidence_resolution"
PREPROCESSING_PLANNING = "03_preprocessing_plan"
SCALE_RELIABILITY = "04_scale_reliability"
MISSINGNESS = "05_missingness"
OUTLIERS = "06_outliers"
DESCRIPTIVE_STATISTICS = "07_descriptive_statistics"
CORRELATION = "08_correlation_analysis"

REGRESSION = "09_regression_analysis"
DIAGNOSTICS = "10_regression_diagnostics"
ROBUSTNESS = "11_robustness_analysis"
ADVANCED_ROBUSTNESS = "12_advanced_robustness"
EFFECT_SIZE = "13_effect_size_analysis"
REPORTING = "14_regression_reporting"
VISUALIZATION = "15_regression_visualization"
RESEARCH_AUDIT = "16_research_audit"


def base_analysis_pipeline() -> list[str]:
    """회귀분석 이전 기본 분석 단계 목록."""
    return [
        DATA_LOADING,
        VARIABLE_DETECTION,
        EVIDENCE_RESOLUTION,
        PREPROCESSING_PLANNING,
        SCALE_RELIABILITY,
        MISSINGNESS,
        OUTLIERS,
        DESCRIPTIVE_STATISTICS,
        CORRELATION,
    ]


def regression_pipeline(
    *,
    diagnostics: bool,
    robustness: bool,
    advanced_robustness: bool,
    effect_size: bool = True,
    reporting: bool = True,
    visualization: bool = True,
    research_audit: bool = True,
) -> list[str]:
    """회귀분석 관련 예상 단계 목록."""
    steps = [REGRESSION]

    if diagnostics:
        steps.append(DIAGNOSTICS)

    if robustness:
        steps.append(ROBUSTNESS)

    if advanced_robustness:
        steps.append(ADVANCED_ROBUSTNESS)

    if effect_size:
        steps.append(EFFECT_SIZE)

    if reporting:
        steps.append(REPORTING)

    if visualization:
        steps.append(VISUALIZATION)

    if research_audit:
        steps.append(RESEARCH_AUDIT)

    return steps


def ols_pipeline(
    *,
    robustness: bool = True,
    advanced_robustness: bool | None = None,
) -> list[str]:
    """OLS 회귀 서브파이프라인."""
    if advanced_robustness is None:
        advanced_robustness = robustness

    return regression_pipeline(
        diagnostics=True,
        robustness=robustness,
        advanced_robustness=advanced_robustness,
    )


def logit_pipeline() -> list[str]:
    """Binary Logit 회귀 서브파이프라인."""
    return regression_pipeline(
        diagnostics=True,
        robustness=False,
        advanced_robustness=False,
    )


def ordered_logit_pipeline() -> list[str]:
    """Ordered Logit 회귀 서브파이프라인."""
    return regression_pipeline(
        diagnostics=True,
        robustness=False,
        advanced_robustness=False,
    )


def count_pipeline() -> list[str]:
    """Count 자동선택 회귀 서브파이프라인."""
    return regression_pipeline(
        diagnostics=True,
        robustness=False,
        advanced_robustness=False,
    )


def poisson_pipeline() -> list[str]:
    """기존 Poisson 테스트 호환용 Count 파이프라인 별칭."""
    return count_pipeline()


def full_ols_pipeline(
    *,
    robustness: bool = True,
    advanced_robustness: bool | None = None,
) -> list[str]:
    """전체 OLS 파이프라인."""
    return base_analysis_pipeline() + ols_pipeline(
        robustness=robustness,
        advanced_robustness=advanced_robustness,
    )


def full_logit_pipeline() -> list[str]:
    """전체 Binary Logit 파이프라인."""
    return base_analysis_pipeline() + logit_pipeline()


def full_ordered_logit_pipeline() -> list[str]:
    """전체 Ordered Logit 파이프라인."""
    return base_analysis_pipeline() + ordered_logit_pipeline()


def full_count_pipeline() -> list[str]:
    """전체 Count 자동선택 회귀 파이프라인."""
    return base_analysis_pipeline() + count_pipeline()


def full_poisson_pipeline() -> list[str]:
    """기존 Poisson E2E 테스트 호환용 전체 Count 파이프라인 별칭."""
    return full_count_pipeline()
