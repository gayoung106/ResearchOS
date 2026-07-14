"""연구분석 프레임워크 기본 진입점."""

from __future__ import annotations

from src.pipeline.stages import PIPELINE_STAGES


def main() -> None:
    """현재 등록된 파이프라인 단계를 출력한다."""
    print("등록된 분석 단계")
    for stage in PIPELINE_STAGES:
        print(f"- {stage}")


if __name__ == "__main__":
    main()
