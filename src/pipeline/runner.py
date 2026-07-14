"""전체 분석 파이프라인 실행기."""

from __future__ import annotations


class PipelineRunner:
    """연구 분석 단계를 순차적으로 실행하는 기본 실행기."""

    def run(self) -> None:
        """전체 파이프라인을 실행한다."""
        raise NotImplementedError
