"""ResearchContext와 PipelineState 저장 위치 관리."""

from __future__ import annotations

from pathlib import Path

from src.common.paths import CONFIG_DIR, RESULT_DIR
from src.pipeline.context import ResearchContext
from src.pipeline.state import PipelineState

CONTEXT_PATH = CONFIG_DIR / "research_context.yaml"
PIPELINE_STATE_PATH = RESULT_DIR / "pipeline_state.json"


def save_research_context(context: ResearchContext) -> Path:
    """기본 경로에 ResearchContext를 저장한다."""
    return context.save_yaml(CONTEXT_PATH)


def load_research_context() -> ResearchContext:
    """기본 경로에서 ResearchContext를 불러온다."""
    return ResearchContext.load_yaml(CONTEXT_PATH)


def save_pipeline_state(state: PipelineState) -> Path:
    """기본 경로에 PipelineState를 저장한다."""
    return state.save_json(PIPELINE_STATE_PATH)


def load_pipeline_state() -> PipelineState:
    """기본 경로에서 PipelineState를 불러온다."""
    return PipelineState.load_json(PIPELINE_STATE_PATH)
