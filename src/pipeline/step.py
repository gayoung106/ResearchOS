"""연구 파이프라인 단계의 공통 인터페이스."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.pipeline.context import ResearchContext


@dataclass(slots=True)
class StepResult:
    """파이프라인 단계 실행 결과."""

    stage_name: str
    success: bool
    output_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class PipelineStep(ABC):
    """모든 파이프라인 단계가 구현해야 하는 기본 인터페이스."""

    name: str
    order: int
    required: bool = True

    def __init__(
        self,
        *,
        name: str,
        order: int,
        required: bool = True,
    ) -> None:
        if not name.strip():
            raise ValueError("단계 이름은 비어 있을 수 없습니다.")

        if order < 0:
            raise ValueError("단계 순서는 0 이상이어야 합니다.")

        self.name = name
        self.order = order
        self.required = required

    def should_run(self, context: ResearchContext) -> bool:
        """현재 연구 맥락에서 이 단계를 실행할지 결정한다."""
        return True

    @abstractmethod
    def run(
        self,
        context: ResearchContext,
        working_directory: Path,
    ) -> StepResult:
        """단계를 실행하고 결과를 반환한다."""
        raise NotImplementedError
