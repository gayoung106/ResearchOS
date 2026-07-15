"""파이프라인 테스트 helper 자체의 동작을 검증한다."""

from pathlib import Path

from src.pipeline.context import ResearchContext
from src.pipeline.orchestrator import ResearchOrchestrator
from src.pipeline.step import PipelineStep, StepResult
from tests.support.assertions import (
    assert_step_order,
    assert_steps_not_registered,
    assert_steps_registered,
)


class DummyStep(PipelineStep):
    """테스트용 파이프라인 단계."""

    def __init__(
        self,
        *,
        name: str,
        order: int,
    ) -> None:
        super().__init__(
            name=name,
            order=order,
            required=False,
        )

    def run(
        self,
        context: ResearchContext,
        working_directory: Path,
    ) -> StepResult:
        """성공 결과만 반환한다."""
        return StepResult(
            stage_name=self.name,
            success=True,
        )


def test_step_assertions_ignore_unrelated_new_steps(
    tmp_path: Path,
) -> None:
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(
            project_name="테스트",
        ),
        working_directory=tmp_path,
    )

    orchestrator.register(
        DummyStep(
            name="11_robustness_analysis",
            order=11,
        )
    )
    orchestrator.register(
        DummyStep(
            name="12_advanced_robustness",
            order=12,
        )
    )
    orchestrator.register(
        DummyStep(
            name="13_effect_size_analysis",
            order=13,
        )
    )
    orchestrator.register(
        DummyStep(
            name="99_future_pipeline_step",
            order=99,
        )
    )

    assert_steps_registered(
        orchestrator,
        "11_robustness_analysis",
        "12_advanced_robustness",
        "13_effect_size_analysis",
    )

    assert_step_order(
        orchestrator,
        before="11_robustness_analysis",
        after="12_advanced_robustness",
    )
    assert_step_order(
        orchestrator,
        before="12_advanced_robustness",
        after="13_effect_size_analysis",
    )

    assert_steps_not_registered(
        orchestrator,
        "10_regression_diagnostics",
    )


def test_step_assertions_use_execution_order_not_registration_order(
    tmp_path: Path,
) -> None:
    orchestrator = ResearchOrchestrator(
        context=ResearchContext(
            project_name="테스트",
        ),
        working_directory=tmp_path,
    )

    orchestrator.register(
        DummyStep(
            name="13_effect_size_analysis",
            order=13,
        )
    )
    orchestrator.register(
        DummyStep(
            name="11_robustness_analysis",
            order=11,
        )
    )
    orchestrator.register(
        DummyStep(
            name="12_advanced_robustness",
            order=12,
        )
    )

    assert_step_order(
        orchestrator,
        before="11_robustness_analysis",
        after="12_advanced_robustness",
    )
    assert_step_order(
        orchestrator,
        before="12_advanced_robustness",
        after="13_effect_size_analysis",
    )
