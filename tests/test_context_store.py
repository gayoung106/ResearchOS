"""ResearchContext와 PipelineState 기본 저장소 테스트."""

from pathlib import Path
from unittest.mock import patch

from src.common import context_store
from src.pipeline.context import ResearchContext
from src.pipeline.state import PipelineState


def test_save_research_context_uses_default_context_path(
    tmp_path: Path,
) -> None:
    context = ResearchContext(
        project_name="컨텍스트 저장 테스트",
    )
    expected_path = tmp_path / "research_context.yaml"

    with (
        patch.object(
            context_store,
            "CONTEXT_PATH",
            expected_path,
        ),
        patch.object(
            ResearchContext,
            "save_yaml",
            return_value=expected_path,
        ) as save_yaml,
    ):
        result = context_store.save_research_context(context)

    assert result == expected_path
    save_yaml.assert_called_once_with(expected_path)


def test_load_research_context_uses_default_context_path(
    tmp_path: Path,
) -> None:
    expected_path = tmp_path / "research_context.yaml"
    expected_context = ResearchContext(
        project_name="컨텍스트 불러오기 테스트",
    )

    with (
        patch.object(
            context_store,
            "CONTEXT_PATH",
            expected_path,
        ),
        patch.object(
            ResearchContext,
            "load_yaml",
            return_value=expected_context,
        ) as load_yaml,
    ):
        result = context_store.load_research_context()

    assert result is expected_context
    load_yaml.assert_called_once_with(expected_path)


def test_save_pipeline_state_uses_default_state_path(
    tmp_path: Path,
) -> None:
    state = PipelineState()
    expected_path = tmp_path / "pipeline_state.json"

    with (
        patch.object(
            context_store,
            "PIPELINE_STATE_PATH",
            expected_path,
        ),
        patch.object(
            PipelineState,
            "save_json",
            return_value=expected_path,
        ) as save_json,
    ):
        result = context_store.save_pipeline_state(state)

    assert result == expected_path
    save_json.assert_called_once_with(expected_path)


def test_load_pipeline_state_uses_default_state_path(
    tmp_path: Path,
) -> None:
    expected_path = tmp_path / "pipeline_state.json"
    expected_state = PipelineState()

    with (
        patch.object(
            context_store,
            "PIPELINE_STATE_PATH",
            expected_path,
        ),
        patch.object(
            PipelineState,
            "load_json",
            return_value=expected_state,
        ) as load_json,
    ):
        result = context_store.load_pipeline_state()

    assert result is expected_state
    load_json.assert_called_once_with(expected_path)
