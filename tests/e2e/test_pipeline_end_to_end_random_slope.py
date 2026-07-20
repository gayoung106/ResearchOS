from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config_models import AnalysisPlan, VariableMap
from src.pipeline.builder import build_default_pipeline
from src.pipeline.context import ResearchContext


def test_pipeline_end_to_end_random_slope(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260720)
    group_count, group_size = 16, 10
    groups = np.repeat(np.arange(group_count), group_size)
    x = rng.normal(size=len(groups))
    intercepts = rng.normal(0, 0.7, group_count)
    slopes = rng.normal(0, 0.4, group_count)
    y = 1.0 + 1.8 * x + intercepts[groups] + slopes[groups] * x + rng.normal(0, 0.35, len(groups))
    rawdata = tmp_path / "rawdata"
    rawdata.mkdir()
    pd.DataFrame({"y": y, "x": x, "group": groups}).to_csv(
        rawdata / "random_slope.csv", index=False
    )

    plan = AnalysisPlan.model_validate(
        {
            "variables": {"dependent": ["y"], "independent": ["x"], "clusters": ["group"]},
            "analyses": {
                "regression": {"enabled": True},
                "multilevel": {
                    "enabled": True,
                    "options": {
                        "group_variable": "group",
                        "random_slope_variable": "x",
                        "optimizer": "lbfgs",
                        "max_iterations": 300,
                    },
                },
                "robustness": {"enabled": False},
            },
        }
    )
    variable_map = VariableMap.model_validate(
        {
            "variables": {
                "y": {"role": "dependent", "measurement_level": "continuous"},
                "x": {"role": "independent", "measurement_level": "continuous"},
                "group": {"role": "cluster", "measurement_level": "nominal"},
            }
        }
    )
    context = ResearchContext(project_name="Random Slope E2E")
    orchestrator, runtime = build_default_pipeline(
        context=context,
        analysis_plan=plan,
        variable_map=variable_map,
        working_directory=tmp_path,
    )
    registration = runtime.get_artifact("regression_registration")
    assert registration.model_type == "mixed_random_slope"
    result = orchestrator.run()
    assert result.success is True
    fitted = runtime.get_artifact("regression_result:main_model")
    assert fitted.model_type == "mixed_random_slope"
    assert fitted.fit_statistics["random_slope_variance"] > 0
    assert "random_intercept_slope_covariance" in fitted.fit_statistics
    diagnostics = runtime.get_artifact("regression_diagnostics:main_model")
    assert "near_zero_slope_variance" in diagnostics.summary
    effects = runtime.get_artifact("effect_size_report:main_model")
    assert effects.model_effects["random_slope_variance"] > 0
    report = runtime.get_artifact("regression_publication_report:main_model")
    assert report.model_type == "mixed_random_slope"
    audit = runtime.get_artifact("research_audit:main_model")
    assert audit.metadata["model_type"] == "mixed_random_slope"
