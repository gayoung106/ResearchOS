"""Research Orchestrator 테스트."""

from pathlib import Path

import pytest

from src.pipeline.context import ResearchContext
from src.pipeline.orchestrator import ResearchOrchestrator
from src.pipeline.registry import StepRegistry
from src.pipeline.state import StageStatus
from src.pipeline.step import PipelineStep, StepResult


class SuccessStep(PipelineStep):
    def __init__(
        self,
        name: str,
        order: int,
    ) -> None:
        super().__init__(
            name=name,
            order=order,
            required=True,
        )

    def run(
        self,
        context: ResearchContext,
        working_directory: Path,
    ) -> StepResult:
        output = working_directory / f"{self.name}.txt"
        output.write_text(
            context.project_name,
            encoding="utf-8",
        )
        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(output)],
        )


class FailureStep(PipelineStep):
    def __init__(self) -> None:
        super().__init__(
            name="failure",
            order=20,
            required=True,
        )

    def run(
        self,
        context: ResearchContext,
        working_directory: Path,
    ) -> StepResult:
        raise RuntimeError("의도된 실패")


class ConditionalStep(SuccessStep):
    def should_run(
        self,
        context: ResearchContext,
    ) -> bool:
        return bool(context.hypotheses)


def context() -> ResearchContext:
    return ResearchContext(
        project_name="테스트 연구",
        research_topic="테스트 주제",
        research_questions=["연구질문"],
    )


def test_registry_orders_steps() -> None:
    registry = StepRegistry()
    registry.register(SuccessStep("second", 20))
    registry.register(SuccessStep("first", 10))

    assert registry.names() == ["first", "second"]


def test_registry_rejects_duplicate_name() -> None:
    registry = StepRegistry()
    registry.register(SuccessStep("same", 10))

    with pytest.raises(ValueError, match="이미 등록된 단계 이름"):
        registry.register(SuccessStep("same", 20))


def test_registry_rejects_duplicate_order() -> None:
    registry = StepRegistry()
    registry.register(SuccessStep("first", 10))

    with pytest.raises(ValueError, match="이미 사용 중인 단계 순서"):
        registry.register(SuccessStep("second", 10))


def test_orchestrator_runs_steps_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    orchestrator = ResearchOrchestrator(
        context=context(),
        working_directory=tmp_path,
    )
    orchestrator.register(SuccessStep("first", 10))
    orchestrator.register(SuccessStep("second", 20))

    result = orchestrator.run()

    assert result.success is True
    assert result.completed_stages == ["first", "second"]
    assert orchestrator.state.stages["first"].status == (StageStatus.COMPLETED)
    assert len(orchestrator.context.generated_files) == 2


def test_orchestrator_stops_on_required_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    orchestrator = ResearchOrchestrator(
        context=context(),
        working_directory=tmp_path,
    )
    orchestrator.register(SuccessStep("first", 10))
    orchestrator.register(FailureStep())
    orchestrator.register(SuccessStep("third", 30))

    result = orchestrator.run()

    assert result.success is False
    assert result.failed_stage == "failure"
    assert "third" not in result.completed_stages
    assert orchestrator.state.stages["failure"].status == (StageStatus.FAILED)


def test_conditional_step_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    orchestrator = ResearchOrchestrator(
        context=context(),
        working_directory=tmp_path,
    )
    orchestrator.register(ConditionalStep("conditional", 10))

    result = orchestrator.run()

    assert result.success is True
    assert result.skipped_stages == ["conditional"]
    assert orchestrator.state.stages["conditional"].status == (StageStatus.SKIPPED)


def test_completed_stage_is_not_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    orchestrator = ResearchOrchestrator(
        context=context(),
        working_directory=tmp_path,
    )
    orchestrator.register(SuccessStep("first", 10))

    first_result = orchestrator.run()
    second_result = orchestrator.run()

    assert first_result.completed_stages == ["first"]
    assert second_result.skipped_stages == ["first"]


def test_start_and_end_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    orchestrator = ResearchOrchestrator(
        context=context(),
        working_directory=tmp_path,
    )
    orchestrator.register(SuccessStep("first", 10))
    orchestrator.register(SuccessStep("second", 20))
    orchestrator.register(SuccessStep("third", 30))

    result = orchestrator.run(
        start_from="second",
        end_at="third",
    )

    assert result.completed_stages == ["second", "third"]
    assert "first" not in orchestrator.context.completed_stages
