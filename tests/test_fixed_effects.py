"""OLS, Binary Logit 및 Ordered Logit 고정효과 처리 테스트."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.common.config_models import (
    AnalysisPlan,
    VariableMap,
)
from src.statistics.regression.binary_logit import (
    fit_binary_logit,
)
from src.statistics.regression.ols import fit_ols
from src.statistics.regression.ordered_logit import (
    fit_ordered_logit,
)
from tests.support.builders import (
    build_regression_pipeline,
)


def fixed_effect_dataframe() -> pd.DataFrame:
    """고정효과 OLS 테스트용 데이터를 생성한다."""
    return pd.DataFrame(
        {
            "y": [
                11,
                13,
                12,
                18,
                19,
                21,
                25,
                24,
                28,
                30,
                31,
                33,
            ],
            "x": list(range(1, 13)),
            "country": [
                "KR",
                "US",
                "JP",
                "KR",
                "US",
                "JP",
                "KR",
                "US",
                "JP",
                "KR",
                "US",
                "JP",
            ],
        }
    )


def binary_fixed_effect_dataframe() -> pd.DataFrame:
    """고정효과 Binary Logit 테스트용 데이터를 생성한다."""
    rows: list[dict[str, Any]] = []

    outcomes = {
        "JP": [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        "KR": [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        "US": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    }

    for country, values in outcomes.items():
        for index, outcome in enumerate(
            values,
            start=1,
        ):
            rows.append(
                {
                    "y": outcome,
                    "x": float(index)
                    + (0.1 if country == "KR" else 0.2 if country == "US" else 0.0),
                    "country": country,
                }
            )

    return pd.DataFrame(rows)


def ordered_fixed_effect_dataframe() -> pd.DataFrame:
    """고정효과 Ordered Logit 테스트용 데이터를 생성한다."""
    rows: list[dict[str, Any]] = []

    outcomes = {
        "JP": [
            1,
            1,
            2,
            2,
            3,
            3,
            4,
            4,
            1,
            2,
            3,
            4,
        ],
        "KR": [
            1,
            2,
            2,
            3,
            3,
            4,
            4,
            1,
            2,
            3,
            4,
            2,
        ],
        "US": [
            2,
            2,
            3,
            3,
            4,
            4,
            1,
            1,
            2,
            3,
            4,
            3,
        ],
    }

    for country, values in outcomes.items():
        for index, outcome in enumerate(
            values,
            start=1,
        ):
            rows.append(
                {
                    "y": outcome,
                    "x": float(index)
                    + (0.1 if country == "KR" else 0.2 if country == "US" else 0.0),
                    "country": country,
                }
            )

    return pd.DataFrame(rows)


def test_ols_encodes_fixed_effects() -> None:
    result = fit_ols(
        fixed_effect_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
        fixed_effects=["country"],
    )

    coefficient_terms = {coefficient.term for coefficient in result.coefficients}

    assert result.model_type == "ols"
    assert result.metadata["fixed_effects"] == [
        "country",
    ]
    assert result.metadata["fixed_effect_reference_categories"] == {
        "country": "JP",
    }
    assert "country_KR" in coefficient_terms
    assert "country_US" in coefficient_terms
    assert "country_JP" not in coefficient_terms


def test_binary_logit_encodes_fixed_effects() -> None:
    result = fit_binary_logit(
        binary_fixed_effect_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
        fixed_effects=["country"],
    )

    coefficient_terms = {coefficient.term for coefficient in result.coefficients}

    assert result.model_type == "binary_logit"
    assert result.converged is True
    assert result.metadata["fixed_effects"] == [
        "country",
    ]
    assert result.metadata["fixed_effect_reference_categories"] == {
        "country": "JP",
    }
    assert "country_KR" in coefficient_terms
    assert "country_US" in coefficient_terms
    assert "country_JP" not in coefficient_terms


def test_ordered_logit_encodes_fixed_effects() -> None:
    result = fit_ordered_logit(
        ordered_fixed_effect_dataframe(),
        dependent_variable="y",
        independent_variables=["x"],
        fixed_effects=["country"],
    )

    coefficient_terms = {coefficient.term for coefficient in result.coefficients}

    assert result.model_type == "ordered_logit"
    assert result.converged is True
    assert result.metadata["fixed_effects"] == [
        "country",
    ]
    assert result.metadata["fixed_effect_reference_categories"] == {
        "country": "JP",
    }
    assert "country_KR" in coefficient_terms
    assert "country_US" in coefficient_terms
    assert "country_JP" not in coefficient_terms


@pytest.mark.parametrize(
    (
        "fit_function",
        "dataframe_factory",
        "expected_columns",
    ),
    [
        (
            fit_ols,
            fixed_effect_dataframe,
            [
                "const",
                "x",
                "country_KR",
                "country_US",
            ],
        ),
        (
            fit_binary_logit,
            binary_fixed_effect_dataframe,
            [
                "const",
                "x",
                "country_KR",
                "country_US",
            ],
        ),
        (
            fit_ordered_logit,
            ordered_fixed_effect_dataframe,
            [
                "x",
                "country_KR",
                "country_US",
            ],
        ),
    ],
)
def test_records_fixed_effect_design_columns(
    fit_function: Callable[..., Any],
    dataframe_factory: Callable[
        [],
        pd.DataFrame,
    ],
    expected_columns: list[str],
) -> None:
    result = fit_function(
        dataframe_factory(),
        dependent_variable="y",
        independent_variables=["x"],
        fixed_effects=["country"],
    )

    assert result.metadata["fixed_effect_columns"] == [
        "country_KR",
        "country_US",
    ]
    assert result.metadata["fixed_effect_column_count"] == 2
    assert result.metadata["design_matrix_columns"] == expected_columns
    assert result.independent_variables == [
        "x",
    ]


@pytest.mark.parametrize(
    (
        "fit_function",
        "dataframe_factory",
    ),
    [
        (
            fit_ols,
            fixed_effect_dataframe,
        ),
        (
            fit_binary_logit,
            binary_fixed_effect_dataframe,
        ),
        (
            fit_ordered_logit,
            ordered_fixed_effect_dataframe,
        ),
    ],
)
def test_missing_fixed_effect_variable_raises(
    fit_function: Callable[..., Any],
    dataframe_factory: Callable[
        [],
        pd.DataFrame,
    ],
) -> None:
    with pytest.raises(
        KeyError,
        match="고정효과 변수가 없습니다",
    ):
        fit_function(
            dataframe_factory(),
            dependent_variable="y",
            independent_variables=["x"],
            fixed_effects=["missing_country"],
        )


@pytest.mark.parametrize(
    (
        "fit_function",
        "dataframe_factory",
    ),
    [
        (
            fit_ols,
            fixed_effect_dataframe,
        ),
        (
            fit_binary_logit,
            binary_fixed_effect_dataframe,
        ),
        (
            fit_ordered_logit,
            ordered_fixed_effect_dataframe,
        ),
    ],
)
def test_constant_fixed_effect_raises(
    fit_function: Callable[..., Any],
    dataframe_factory: Callable[
        [],
        pd.DataFrame,
    ],
) -> None:
    dataframe = dataframe_factory()
    dataframe["country"] = "KR"

    with pytest.raises(
        ValueError,
        match="유효 범주가 하나뿐",
    ):
        fit_function(
            dataframe,
            dependent_variable="y",
            independent_variables=["x"],
            fixed_effects=["country"],
        )


@pytest.mark.parametrize(
    (
        "fit_function",
        "dataframe_factory",
    ),
    [
        (
            fit_ols,
            fixed_effect_dataframe,
        ),
        (
            fit_binary_logit,
            binary_fixed_effect_dataframe,
        ),
        (
            fit_ordered_logit,
            ordered_fixed_effect_dataframe,
        ),
    ],
)
def test_duplicate_predictor_and_fixed_effect_raises(
    fit_function: Callable[..., Any],
    dataframe_factory: Callable[
        [],
        pd.DataFrame,
    ],
) -> None:
    with pytest.raises(
        ValueError,
        match="중복 지정",
    ):
        fit_function(
            dataframe_factory(),
            dependent_variable="y",
            independent_variables=[
                "x",
                "country",
            ],
            fixed_effects=["country"],
        )


@pytest.mark.parametrize(
    (
        "fit_function",
        "dataframe_factory",
        "expected_sample_size",
    ),
    [
        (
            fit_ols,
            fixed_effect_dataframe,
            11,
        ),
        (
            fit_binary_logit,
            binary_fixed_effect_dataframe,
            35,
        ),
        (
            fit_ordered_logit,
            ordered_fixed_effect_dataframe,
            35,
        ),
    ],
)
def test_missing_fixed_effect_cases_are_dropped(
    fit_function: Callable[..., Any],
    dataframe_factory: Callable[
        [],
        pd.DataFrame,
    ],
    expected_sample_size: int,
) -> None:
    dataframe = dataframe_factory()
    dataframe.loc[0, "country"] = None

    result = fit_function(
        dataframe,
        dependent_variable="y",
        independent_variables=["x"],
        fixed_effects=["country"],
    )

    assert result.sample_size == (expected_sample_size)
    assert result.metadata["dropped_case_count"] == 1


def _build_fixed_effect_plan(
    measurement_level: str,
) -> tuple[
    AnalysisPlan,
    VariableMap,
]:
    analysis_plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x"],
                "fixed_effects": ["country"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                },
                "robustness": {
                    "enabled": False,
                },
            },
        }
    )

    variable_map = VariableMap.model_validate(
        {
            "variables": {
                "y": {
                    "role": "dependent",
                    "measurement_level": (measurement_level),
                },
                "x": {
                    "role": "independent",
                    "measurement_level": ("continuous"),
                },
                "country": {
                    "role": "fixed_effect",
                    "measurement_level": ("nominal"),
                },
            }
        }
    )

    return analysis_plan, variable_map


@pytest.mark.parametrize(
    (
        "measurement_level",
        "expected_model_type",
    ),
    [
        (
            "continuous",
            "ols",
        ),
        (
            "binary",
            "binary_logit",
        ),
        (
            "ordinal",
            "ordered_logit",
        ),
    ],
)
def test_builder_registers_fixed_effects(
    tmp_path: Path,
    measurement_level: str,
    expected_model_type: str,
) -> None:
    analysis_plan, variable_map = _build_fixed_effect_plan(measurement_level)

    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=analysis_plan,
        variable_map=variable_map,
    )

    regression_step = orchestrator.registry.get("09_regression_analysis")

    assert registration.registered is True
    assert registration.model_type == (expected_model_type)
    assert registration.independent_variables == [
        "x",
    ]
    assert registration.fixed_effects == [
        "country",
    ]
    assert regression_step.independent_variables == [
        "x",
    ]
    assert regression_step.fixed_effects == [
        "country",
    ]

    assert registration.diagnostics_registered is True


def test_builder_rejects_missing_fixed_effect_definition(
    tmp_path: Path,
) -> None:
    analysis_plan = AnalysisPlan.model_validate(
        {
            "variables": {
                "dependent": ["y"],
                "independent": ["x"],
                "fixed_effects": ["country"],
            },
            "analyses": {
                "regression": {
                    "enabled": True,
                }
            },
        }
    )
    variable_map = VariableMap.model_validate(
        {
            "variables": {
                "y": {
                    "measurement_level": ("continuous"),
                },
                "x": {
                    "measurement_level": ("continuous"),
                },
            }
        }
    )

    orchestrator, _, registration = build_regression_pipeline(
        tmp_path,
        analysis_plan=analysis_plan,
        variable_map=variable_map,
    )

    assert registration.registered is False
    assert registration.fixed_effects == [
        "country",
    ]
    assert "variable_map 정의가 없습니다" in registration.warnings[0]
    assert orchestrator.registry.names() == []
