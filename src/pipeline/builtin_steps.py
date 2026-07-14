"""프레임워크 점검용 기본 파이프라인 단계."""

from __future__ import annotations

from pathlib import Path

from src.pipeline.context import ResearchContext
from src.pipeline.step import PipelineStep, StepResult


class ProjectValidationStep(PipelineStep):
    """프로젝트 기본정보를 검증하는 단계."""

    def __init__(self) -> None:
        super().__init__(
            name="00_project_validation",
            order=0,
            required=True,
        )

    def run(
        self,
        context: ResearchContext,
        working_directory: Path,
    ) -> StepResult:
        warnings: list[str] = []

        if not context.research_topic.strip():
            warnings.append("연구주제가 비어 있습니다.")

        if not context.research_questions:
            warnings.append("연구질문이 등록되지 않았습니다.")

        return StepResult(
            stage_name=self.name,
            success=True,
            warnings=warnings,
            metadata={
                "project_name": context.project_name,
                "working_directory": str(working_directory),
            },
        )


class DirectoryPreparationStep(PipelineStep):
    """결과 디렉터리를 준비하는 단계."""

    def __init__(self) -> None:
        super().__init__(
            name="01_directory_preparation",
            order=10,
            required=True,
        )

    def run(
        self,
        context: ResearchContext,
        working_directory: Path,
    ) -> StepResult:
        result_directory = working_directory / "result"
        result_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        marker = result_directory / ".pipeline_ready"
        marker.write_text(
            context.project_name,
            encoding="utf-8",
        )

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(marker)],
        )
