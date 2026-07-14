"""파이프라인 단계 등록 및 조회 모듈."""

from __future__ import annotations

from src.pipeline.step import PipelineStep


class StepRegistry:
    """파이프라인 단계를 순서대로 관리한다."""

    def __init__(self) -> None:
        self._steps: dict[str, PipelineStep] = {}

    def register(self, step: PipelineStep) -> None:
        """단계를 등록한다."""
        if step.name in self._steps:
            raise ValueError(f"이미 등록된 단계 이름입니다: {step.name}")

        if any(registered.order == step.order for registered in self._steps.values()):
            raise ValueError(f"이미 사용 중인 단계 순서입니다: {step.order}")

        self._steps[step.name] = step

    def unregister(self, stage_name: str) -> None:
        """등록된 단계를 제거한다."""
        if stage_name not in self._steps:
            raise KeyError(f"등록되지 않은 단계입니다: {stage_name}")

        del self._steps[stage_name]

    def get(self, stage_name: str) -> PipelineStep:
        """단계 이름으로 등록된 단계를 반환한다."""
        try:
            return self._steps[stage_name]
        except KeyError as error:
            raise KeyError(f"등록되지 않은 단계입니다: {stage_name}") from error

    def ordered_steps(self) -> list[PipelineStep]:
        """실행 순서대로 정렬된 단계를 반환한다."""
        return sorted(
            self._steps.values(),
            key=lambda step: step.order,
        )

    def names(self) -> list[str]:
        """실행 순서대로 단계 이름 목록을 반환한다."""
        return [step.name for step in self.ordered_steps()]

    def __len__(self) -> int:
        return len(self._steps)
