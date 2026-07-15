from __future__ import annotations

REGRESSION = "09_regression_analysis"
DIAGNOSTICS = "10_regression_diagnostics"
ROBUSTNESS = "11_robustness_analysis"
ADVANCED_ROBUSTNESS = "12_advanced_robustness"
EFFECT_SIZE = "13_effect_size_analysis"
REPORTING = "14_regression_reporting"
VISUALIZATION = "15_regression_visualization"
RESEARCH_AUDIT = "16_research_audit"


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
    if advanced_robustness is None:
        advanced_robustness = robustness

    return regression_pipeline(
        diagnostics=True,
        robustness=robustness,
        advanced_robustness=advanced_robustness,
    )


def logit_pipeline() -> list[str]:
    return regression_pipeline(
        diagnostics=False,
        robustness=False,
        advanced_robustness=False,
    )
