from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.builder import build_default_pipeline
from src.pipeline.context import ResearchContext
from tests.support.assertions import assert_registry_matches
from tests.support.expected_pipeline import full_count_pipeline


def test_pipeline_end_to_end_zero_inflated_count(
    tmp_path: Path,
    ols_with_robustness_analysis_plan,
    count_variable_map,
) -> None:
    rng = np.random.default_rng(20260716)
    size = 700
    x = rng.normal(size=size)
    mean = np.exp(0.4 + 0.5 * x)
    alpha = 1.2
    multiplier = rng.gamma(
        shape=1 / alpha,
        scale=alpha,
        size=size,
    )
    y = rng.poisson(mean * multiplier)
    y[rng.random(size) < 0.40] = 0

    rawdata_dir = tmp_path / "rawdata"
    rawdata_dir.mkdir()

    pd.DataFrame(
        {
            "y": y,
            "x": x,
        }
    ).to_csv(
        rawdata_dir / "zero_inflated_count.csv",
        index=False,
    )

    orchestrator, runtime = build_default_pipeline(
        context=ResearchContext(
            project_name="Zero Inflated Count E2E",
        ),
        analysis_plan=ols_with_robustness_analysis_plan,
        variable_map=count_variable_map,
        working_directory=tmp_path,
    )

    assert_registry_matches(
        orchestrator,
        full_count_pipeline(),
    )

    pipeline_result = orchestrator.run()

    assert pipeline_result.success is True
    assert pipeline_result.failed_stage is None

    regression_result = runtime.get_artifact("regression_result:main_model")
    diagnostics = runtime.get_artifact("regression_diagnostics:main_model")
    effect_report = runtime.get_artifact("effect_size_report:main_model")

    assert regression_result.model_type in {
        "zero_inflated_poisson",
        "zero_inflated_negative_binomial",
    }
    assert diagnostics.model_type == regression_result.model_type
    assert diagnostics.sample_size == size
    assert any(effect.effect_type == "incidence_rate_ratio" for effect in effect_report.effects)
