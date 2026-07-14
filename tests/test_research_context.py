"""ResearchContext 및 PipelineState 테스트."""

from pathlib import Path

from src.pipeline.context import ResearchContext
from src.pipeline.state import PipelineState, StageStatus


def test_research_context_round_trip(tmp_path: Path) -> None:
    context = ResearchContext(
        project_name="테스트 연구",
        research_topic="공공부문 데이터 분석",
        research_questions=["공공부문과 민간부문의 차이가 있는가?"],
    )
    context.mark_stage_completed("01_conversion")
    context.add_warning("코드북 확인 필요")

    output_path = tmp_path / "research_context.yaml"
    context.save_yaml(output_path)

    loaded = ResearchContext.load_yaml(output_path)

    assert loaded.project_name == "테스트 연구"
    assert loaded.research_topic == "공공부문 데이터 분석"
    assert loaded.research_questions == ["공공부문과 민간부문의 차이가 있는가?"]
    assert loaded.completed_stages == ["01_conversion"]
    assert loaded.warnings == ["코드북 확인 필요"]


def test_pipeline_state_round_trip(tmp_path: Path) -> None:
    state = PipelineState()
    state.start_stage("01_conversion")
    state.complete_stage(
        "01_conversion",
        output_files=["result/01_conversion/data.parquet"],
    )

    output_path = tmp_path / "pipeline_state.json"
    state.save_json(output_path)

    loaded = PipelineState.load_json(output_path)
    record = loaded.stages["01_conversion"]

    assert loaded.active_stage is None
    assert record.status == StageStatus.COMPLETED
    assert record.output_files == ["result/01_conversion/data.parquet"]


def test_pipeline_failure_state() -> None:
    state = PipelineState()
    state.start_stage("02_diagnostics")
    state.fail_stage("02_diagnostics", "진단 실패")

    record = state.stages["02_diagnostics"]

    assert state.active_stage is None
    assert record.status == StageStatus.FAILED
    assert record.error_message == "진단 실패"
