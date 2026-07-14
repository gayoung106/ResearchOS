"""설정 파일을 읽고 기본 연구 파이프라인을 실행하는 진입점."""

from __future__ import annotations

from src.common.config_loader import (
    build_research_context,
    load_analysis_plan,
    load_research_plan,
    load_variable_map,
)
from src.common.context_store import save_research_context
from src.common.paths import CONFIG_DIR, PROJECT_ROOT
from src.pipeline.builder import build_default_pipeline


def main() -> None:
    """설정 파일을 읽고 기본 파이프라인을 실행한다."""
    research_plan = load_research_plan(CONFIG_DIR / "research_plan.yaml")
    analysis_plan = load_analysis_plan(CONFIG_DIR / "analysis_plan.yaml")
    variable_map = load_variable_map(CONFIG_DIR / "variable_map.yaml")

    context = build_research_context(
        research_plan,
        analysis_plan,
        variable_map,
    )
    save_research_context(context)

    orchestrator, _ = build_default_pipeline(
        context=context,
        analysis_plan=analysis_plan,
        variable_map=variable_map,
        working_directory=PROJECT_ROOT,
    )

    result = orchestrator.run()

    if result.success:
        print("연구 파이프라인이 정상 완료되었습니다.")
    else:
        print(f"연구 파이프라인이 실패했습니다: {result.failed_stage}")


if __name__ == "__main__":
    main()
