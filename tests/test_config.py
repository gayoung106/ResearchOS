"""연구 설정 로더와 검증기 테스트."""

from pathlib import Path

import pytest
import yaml

from src.common.config_exceptions import ConfigValidationError
from src.common.config_loader import (
    build_research_context,
    load_analysis_plan,
    load_research_plan,
    load_variable_map,
)


def write_yaml(path: Path, data: dict) -> None:
    """테스트용 YAML 파일을 저장한다."""
    path.write_text(
        yaml.safe_dump(
            data,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_load_valid_configs_and_build_context(tmp_path: Path) -> None:
    research_path = tmp_path / "research_plan.yaml"
    analysis_path = tmp_path / "analysis_plan.yaml"
    variable_path = tmp_path / "variable_map.yaml"

    write_yaml(
        research_path,
        {
            "project": {"title": "테스트 연구"},
            "research": {
                "topic": "공공부문과 민간부문의 차이",
                "research_questions": ["두 집단에 차이가 있는가?"],
            },
            "design": {},
            "data": {},
            "review": {},
        },
    )
    write_yaml(
        analysis_path,
        {
            "variables": {
                "dependent": ["outcome"],
                "independent": ["public_sector"],
            },
            "preprocessing": {},
            "analyses": {
                "descriptive": {"enabled": True},
                "reliability": {},
                "validity": {},
                "regression": {"enabled": True},
                "mediation": {},
                "moderation": {},
                "multilevel": {},
                "panel": {},
                "robustness": {"enabled": True},
            },
            "outputs": {},
            "review": {},
        },
    )
    write_yaml(
        variable_path,
        {
            "variables": {
                "outcome": {
                    "original_name": "outcome",
                    "role": "dependent",
                    "measurement_level": "continuous",
                },
                "public_sector": {
                    "original_name": "public_sector",
                    "role": "independent",
                    "measurement_level": "binary",
                },
            }
        },
    )

    research_plan = load_research_plan(research_path)
    analysis_plan = load_analysis_plan(analysis_path)
    variable_map = load_variable_map(variable_path)

    context = build_research_context(
        research_plan,
        analysis_plan,
        variable_map,
    )

    assert context.project_name == "테스트 연구"
    assert context.dependent_variables == ["outcome"]
    assert context.independent_variables == ["public_sector"]
    assert context.warnings == []


def test_duplicate_variable_roles_raise_error(tmp_path: Path) -> None:
    analysis_path = tmp_path / "analysis_plan.yaml"

    write_yaml(
        analysis_path,
        {
            "variables": {
                "dependent": ["same_variable"],
                "independent": ["same_variable"],
            },
            "preprocessing": {},
            "analyses": {},
            "outputs": {},
            "review": {},
        },
    )

    with pytest.raises(ConfigValidationError):
        load_analysis_plan(analysis_path)


def test_regression_requires_dependent_variable(tmp_path: Path) -> None:
    research_path = tmp_path / "research_plan.yaml"
    analysis_path = tmp_path / "analysis_plan.yaml"
    variable_path = tmp_path / "variable_map.yaml"

    write_yaml(
        research_path,
        {
            "project": {"title": "테스트 연구"},
            "research": {"topic": "테스트"},
            "design": {},
            "data": {},
            "review": {},
        },
    )
    write_yaml(
        analysis_path,
        {
            "variables": {},
            "preprocessing": {},
            "analyses": {
                "regression": {"enabled": True},
            },
            "outputs": {},
            "review": {},
        },
    )
    write_yaml(variable_path, {"variables": {}})

    research_plan = load_research_plan(research_path)
    analysis_plan = load_analysis_plan(analysis_path)
    variable_map = load_variable_map(variable_path)

    with pytest.raises(
        ConfigValidationError,
        match="종속변수가 지정되지 않았습니다",
    ):
        build_research_context(
            research_plan,
            analysis_plan,
            variable_map,
        )


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    research_path = tmp_path / "research_plan.yaml"

    write_yaml(
        research_path,
        {
            "project": {
                "title": "테스트 연구",
                "unknown_key": "허용되지 않음",
            },
            "research": {},
            "design": {},
            "data": {},
            "review": {},
        },
    )

    with pytest.raises(ConfigValidationError):
        load_research_plan(research_path)
