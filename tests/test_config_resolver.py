"""변수명 자동해결기 테스트."""

import pandas as pd
import pytest

from src.common.config_models import AnalysisPlan
from src.common.config_resolver import (
    apply_confirmed_resolutions,
    resolution_summary,
    resolve_analysis_variables,
    resolve_variable_name,
    validate_confirmed_mapping,
)


def test_exact_variable_match() -> None:
    result = resolve_variable_name(
        "public_sector",
        ["public_sector", "age"],
    )

    assert result.status == "resolved"
    assert result.resolved_name == "public_sector"
    assert result.match_type == "exact"


def test_normalized_variable_match() -> None:
    result = resolve_variable_name(
        "public-sector",
        ["public_sector", "age"],
    )

    assert result.status == "resolved"
    assert result.resolved_name == "public_sector"
    assert result.match_type == "normalized"


def test_similar_variable_requires_review() -> None:
    result = resolve_variable_name(
        "job_satisfaction",
        ["job_sat", "age", "income"],
        similarity_threshold=0.5,
    )

    assert result.status == "review_required"
    assert result.resolved_name is None
    assert result.candidates[0].candidate_name == "job_sat"


def test_variable_label_can_produce_candidate() -> None:
    metadata = pd.DataFrame(
        {
            "variable_name": ["q01", "q02"],
            "variable_label": ["직무 만족도", "조직 몰입"],
        }
    )
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["직무만족도"],
            }
        }
    )

    resolutions = resolve_analysis_variables(
        plan,
        ["q01", "q02"],
        variable_metadata=metadata,
        similarity_threshold=0.7,
    )

    assert resolutions[0].status == "review_required"
    assert resolutions[0].candidates[0].candidate_name == "q01"


def test_apply_confirmed_resolution() -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["job_satisfaction"],
                "independent": ["public_sector"],
            }
        }
    )

    updated = apply_confirmed_resolutions(
        plan,
        {"job_satisfaction": "job_sat"},
    )

    assert updated.variables.dependent == ["job_sat"]
    assert updated.variables.independent == ["public_sector"]


def test_resolution_summary() -> None:
    plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["outcome"],
                "independent": ["public-sector"],
                "controls": ["unknown_variable"],
            }
        }
    )

    resolutions = resolve_analysis_variables(
        plan,
        ["outcome", "public_sector"],
        similarity_threshold=0.9,
    )
    summary = resolution_summary(resolutions)

    assert summary["resolved"] == 2
    assert summary["not_found"] == 1


def test_invalid_confirmed_mapping_raises_error() -> None:
    with pytest.raises(ValueError, match="실제 데이터에 없습니다"):
        validate_confirmed_mapping(
            {"job_satisfaction": "missing_column"},
            ["job_sat", "age"],
        )
