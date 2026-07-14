"""연구분석 파이프라인 전체를 조정하는 오케스트레이터."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.common.context_store import (
    save_pipeline_state,
    save_research_context,
)
from src.common.logger import setup_logger
from src.pipeline.context import ResearchContext
from src.pipeline.registry import StepRegistry
from src.pipeline.state import PipelineState, StageStatus
from src.pipeline.step import PipelineStep, StepResult


@dataclass(slots=True)
class OrchestratorResult:
    """전체 파이프라인 실행 결과."""

    success: bool
    completed_stages: list[str] = field(default_factory=list)
    failed_stage: str | None = None
    skipped_stages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ResearchOrchestrator:
    """연구분석 단계를 순서대로 실행하고 상태를 저장한다."""

    def __init__(
        self,
        *,
        context: ResearchContext,
        registry: StepRegistry | None = None,
        state: PipelineState | None = None,
        working_directory: str | Path = ".",
        stop_on_failure: bool = True,
    ) -> None:
        self.context = context
        self.registry = registry or StepRegistry()
        self.state = state or PipelineState()
        self.working_directory = Path(working_directory).expanduser().resolve()
        self.stop_on_failure = stop_on_failure
        self.logger = setup_logger(
            "research_orchestrator",
            "research_orchestrator.log",
        )

    def register(self, step: PipelineStep) -> None:
        """파이프라인 단계를 등록한다."""
        self.registry.register(step)
        self.state.register_stage(step.name)

    def run(
        self,
        *,
        start_from: str | None = None,
        end_at: str | None = None,
        rerun_completed: bool = False,
    ) -> OrchestratorResult:
        """
        등록된 단계를 순서대로 실행한다.

        Args:
            start_from: 이 단계부터 실행
            end_at: 이 단계까지 실행
            rerun_completed: 이미 완료된 단계를 다시 실행할지 여부
        """
        steps = self._select_steps(
            start_from=start_from,
            end_at=end_at,
        )

        completed_stages: list[str] = []
        skipped_stages: list[str] = []
        warnings: list[str] = []

        self.logger.info(
            "연구 파이프라인을 시작합니다. 등록 단계 수: %s",
            len(steps),
        )

        for step in steps:
            current_record = self.state.stages.get(step.name)

            if (
                current_record
                and current_record.status == StageStatus.COMPLETED
                and not rerun_completed
            ):
                self.logger.info(
                    "이미 완료된 단계를 생략합니다: %s",
                    step.name,
                )
                skipped_stages.append(step.name)
                continue

            if not step.should_run(self.context):
                self.logger.info(
                    "실행 조건을 충족하지 않아 생략합니다: %s",
                    step.name,
                )
                self.state.skip_stage(step.name)
                skipped_stages.append(step.name)
                self._save_checkpoint()
                continue

            self.logger.info("단계 시작: %s", step.name)
            self.state.start_stage(step.name)
            self._save_checkpoint()

            try:
                result = step.run(
                    self.context,
                    self.working_directory,
                )
            except Exception as error:
                self.logger.exception(
                    "단계 실행 중 예외가 발생했습니다: %s",
                    step.name,
                )
                self.state.fail_stage(
                    step.name,
                    str(error),
                )
                self.context.add_warning(f"{step.name} 실행 실패: {error}")
                self._save_checkpoint()

                if self.stop_on_failure or step.required:
                    return OrchestratorResult(
                        success=False,
                        completed_stages=completed_stages,
                        failed_stage=step.name,
                        skipped_stages=skipped_stages,
                        warnings=warnings + [str(error)],
                    )

                warnings.append(str(error))
                continue

            if not result.success:
                message = (
                    result.metadata.get("error_message")
                    or f"{step.name} 단계가 실패 결과를 반환했습니다."
                )
                self.state.fail_stage(
                    step.name,
                    str(message),
                )
                warnings.extend(result.warnings)
                self.context.warnings.extend(result.warnings)
                self._save_checkpoint()

                if self.stop_on_failure or step.required:
                    return OrchestratorResult(
                        success=False,
                        completed_stages=completed_stages,
                        failed_stage=step.name,
                        skipped_stages=skipped_stages,
                        warnings=warnings,
                    )

                continue

            self._apply_step_result(result)
            self.state.complete_stage(
                step.name,
                output_files=result.output_files,
            )
            self.context.mark_stage_completed(step.name)
            completed_stages.append(step.name)
            warnings.extend(result.warnings)

            self.logger.info("단계 완료: %s", step.name)
            self._save_checkpoint()

        self.logger.info("연구 파이프라인이 완료되었습니다.")

        return OrchestratorResult(
            success=True,
            completed_stages=completed_stages,
            skipped_stages=skipped_stages,
            warnings=warnings,
        )

    def _apply_step_result(self, result: StepResult) -> None:
        """단계 결과를 ResearchContext에 반영한다."""
        for output_file in result.output_files:
            self.context.add_generated_file(output_file)

        for warning in result.warnings:
            self.context.add_warning(f"{result.stage_name}: {warning}")

    def _select_steps(
        self,
        *,
        start_from: str | None,
        end_at: str | None,
    ) -> list[PipelineStep]:
        """실행 범위에 해당하는 단계만 반환한다."""
        steps = self.registry.ordered_steps()
        names = [step.name for step in steps]

        if start_from is not None and start_from not in names:
            raise ValueError(f"start_from 단계가 등록되어 있지 않습니다: {start_from}")

        if end_at is not None and end_at not in names:
            raise ValueError(f"end_at 단계가 등록되어 있지 않습니다: {end_at}")

        start_index = names.index(start_from) if start_from is not None else 0
        end_index = names.index(end_at) + 1 if end_at is not None else len(steps)

        if start_index >= end_index:
            raise ValueError("start_from은 end_at보다 앞선 단계여야 합니다.")

        return steps[start_index:end_index]

    def _save_checkpoint(self) -> None:
        """현재 연구 맥락과 파이프라인 상태를 저장한다."""
        save_research_context(self.context)
        save_pipeline_state(self.state)

    def status_summary(self) -> dict[str, Any]:
        """현재 단계별 상태를 요약한다."""
        return {
            "project_name": self.context.project_name,
            "registered_stages": self.registry.names(),
            "active_stage": self.state.active_stage,
            "stages": {name: record.status.value for name, record in self.state.stages.items()},
        }
