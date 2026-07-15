"""전체 연구 파이프라인 조립 테스트."""

from pathlib import Path

from src.common.config_models import AnalysisPlan, VariableMap
from src.pipeline.builder import build_default_pipeline
from src.pipeline.context import ResearchContext
from tests.support.assertions import assert_registry_matches
from tests.support.expected_pipeline import (
    full_logit_pipeline,
    full_ols_pipeline,
)


def test_default_builder_registers_full_ols_pipeline(
    tmp_path: Path,
    ols_with_robustness_analysis_plan: AnalysisPlan,
    continuous_variable_map: VariableMap,
) -> None:
    context = ResearchContext(
        project_name="전체 OLS 파이프라인 테스트",
    )

    orchestrator, runtime = build_default_pipeline(
        context=context,
        analysis_plan=ols_with_robustness_analysis_plan,
        variable_map=continuous_variable_map,
        working_directory=tmp_path,
    )

    assert_registry_matches(
        orchestrator,
        full_ols_pipeline(
            robustness=True,
            advanced_robustness=True,
        ),
    )

    registration = runtime.get_artifact(
        "regression_registration",
    )

    assert registration.registered is True
    assert registration.model_type == "ols"
    assert registration.audit_registered is True


def test_default_builder_registers_full_logit_pipeline(
    tmp_path: Path,
    ols_with_robustness_analysis_plan: AnalysisPlan,
    binary_variable_map: VariableMap,
) -> None:
    context = ResearchContext(
        project_name="전체 Logit 파이프라인 테스트",
    )

    orchestrator, runtime = build_default_pipeline(
        context=context,
        analysis_plan=ols_with_robustness_analysis_plan,
        variable_map=binary_variable_map,
        working_directory=tmp_path,
    )

    assert_registry_matches(
        orchestrator,
        full_logit_pipeline(),
    )

    registration = runtime.get_artifact(
        "regression_registration",
    )

    assert registration.registered is True
    assert registration.model_type == "binary_logit"
    assert registration.diagnostics_registered is False
    assert registration.robustness_registered is False
