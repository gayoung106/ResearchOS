"""척도 구성 및 신뢰도 분석 테스트."""

import pandas as pd
import pytest

from src.common.config_models import VariableMap
from src.preprocess.scales import (
    ScaleDefinition,
    build_all_scales,
    build_scale,
    collect_scale_definitions,
)
from src.statistics.reliability import (
    alpha_if_item_deleted,
    corrected_item_total_correlations,
    cronbach_alpha,
    run_reliability_analysis,
)


def test_collect_scale_definitions() -> None:
    variable_map = VariableMap.model_validate(
        {
            "variables": {
                "q1": {
                    "scale_name": "trust",
                    "reverse_coded": False,
                    "measurement_level": "scale_item",
                },
                "q2": {
                    "scale_name": "trust",
                    "reverse_coded": True,
                    "measurement_level": "scale_item",
                },
            }
        }
    )

    definitions = collect_scale_definitions(variable_map)

    assert len(definitions) == 1
    assert definitions[0].scale_name == "trust"
    assert definitions[0].items == ["q1", "q2"]
    assert definitions[0].reverse_items == ["q2"]


def test_build_mean_scale_with_minimum_items() -> None:
    dataframe = pd.DataFrame(
        {
            "q1": [1, 2, None],
            "q2": [3, 4, 5],
            "q3": [5, None, 1],
        }
    )
    definition = ScaleDefinition(
        scale_name="trust",
        items=["q1", "q2", "q3"],
        aggregation="mean",
        minimum_valid_items=2,
        output_name="trust_mean",
    )

    scale, record = build_scale(dataframe, definition)

    assert scale.tolist() == [3.0, 3.0, 3.0]
    assert record.valid_case_count == 3


def test_build_all_scales_does_not_modify_original() -> None:
    original = pd.DataFrame(
        {
            "q1": [1, 2],
            "q2": [3, 4],
        }
    )
    definition = ScaleDefinition(
        scale_name="trust",
        items=["q1", "q2"],
        output_name="trust_mean",
    )

    output, records = build_all_scales(
        original,
        [definition],
    )

    assert "trust_mean" not in original.columns
    assert "trust_mean" in output.columns
    assert len(records) == 1


def test_cronbach_alpha_perfect_items() -> None:
    dataframe = pd.DataFrame(
        {
            "q1": [1, 2, 3, 4],
            "q2": [1, 2, 3, 4],
            "q3": [1, 2, 3, 4],
        }
    )

    alpha = cronbach_alpha(dataframe)

    assert alpha == pytest.approx(1.0)


def test_corrected_item_total_correlations() -> None:
    dataframe = pd.DataFrame(
        {
            "q1": [1, 2, 3, 4],
            "q2": [1, 2, 3, 4],
            "q3": [1, 2, 3, 4],
        }
    )

    correlations = corrected_item_total_correlations(dataframe)

    assert correlations["q1"] == pytest.approx(1.0)
    assert correlations["q2"] == pytest.approx(1.0)


def test_alpha_if_item_deleted() -> None:
    dataframe = pd.DataFrame(
        {
            "q1": [1, 2, 3, 4],
            "q2": [1, 2, 3, 4],
            "q3": [1, 2, 3, 4],
        }
    )

    deleted = alpha_if_item_deleted(dataframe)

    assert deleted["q1"] == pytest.approx(1.0)
    assert deleted["q2"] == pytest.approx(1.0)


def test_run_reliability_analysis() -> None:
    dataframe = pd.DataFrame(
        {
            "q1": [1, 2, 3, 4, 5],
            "q2": [1, 2, 3, 4, 5],
            "q3": [2, 2, 3, 4, 4],
        }
    )

    result, item_table = run_reliability_analysis(
        dataframe,
        scale_name="trust",
    )

    assert result.scale_name == "trust"
    assert result.item_count == 3
    assert result.cronbach_alpha is not None
    assert len(item_table) == 3


def test_scale_requires_two_items() -> None:
    dataframe = pd.DataFrame({"q1": [1, 2, 3]})
    definition = ScaleDefinition(
        scale_name="single",
        items=["q1"],
    )

    with pytest.raises(ValueError, match="최소 2개 문항"):
        build_scale(dataframe, definition)
