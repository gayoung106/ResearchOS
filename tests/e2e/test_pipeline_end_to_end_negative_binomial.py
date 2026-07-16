from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.builder import build_default_pipeline
from src.pipeline.context import ResearchContext
from tests.support.assertions import assert_registry_matches
from tests.support.expected_pipeline import full_count_pipeline


def test_pipeline_end_to_end_negative_binomial(
    tmp_path: Path, ols_with_robustness_analysis_plan, count_variable_map
) -> None:
    rng = np.random.default_rng(20260716)
    x = rng.normal(size=400)
    mean = np.exp(0.3 + 0.6 * x)
    alpha = 1.5
    mult = rng.gamma(shape=1 / alpha, scale=alpha, size=len(x))
    y = rng.poisson(mean * mult)
    raw = tmp_path / "rawdata"
    raw.mkdir()
    pd.DataFrame({"y": y, "x": x}).to_csv(raw / "negative_binomial_sample.csv", index=False)
    orchestrator, runtime = build_default_pipeline(
        context=ResearchContext(project_name="Negative Binomial E2E"),
        analysis_plan=ols_with_robustness_analysis_plan,
        variable_map=count_variable_map,
        working_directory=tmp_path,
    )
    assert_registry_matches(orchestrator, full_count_pipeline())
    result = orchestrator.run()
    assert result.success is True
    regression = runtime.get_artifact("regression_result:main_model")
    effects = runtime.get_artifact("effect_size_report:main_model")
    assert regression.model_type == "negative_binomial"
    assert regression.metadata["selected_count_model"] == "negative_binomial"
    assert regression.metadata["poisson_dispersion_ratio"] > 1.5
    assert regression.fit_statistics["alpha"] > 0
    assert any(e.effect_type == "incidence_rate_ratio" for e in effects.effects)
