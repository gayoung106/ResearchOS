"""Automatic project execution helpers."""

from src.auto.analysis_plan import (
    AutoAnalysisPlanDecision,
    AutoAnalysisPlanResult,
    AutoAnalysisPlanStep,
    auto_analysis_plan_decisions_to_dataframe,
    auto_analysis_plan_summary_to_dataframe,
    build_auto_analysis_plan,
    write_auto_analysis_config_files,
)
from src.auto.overrides import (
    AutoVariableRoleOverrides,
    apply_variable_role_overrides,
    build_auto_variable_role_overrides,
)
from src.auto.pipeline import (
    AutoRegressionPipelineBuildResult,
    build_auto_regression_orchestrator,
    register_auto_regression_pipeline,
)
from src.auto.rawdata_loader import (
    AutoRawDataLoadingStep,
    AutoRawDataLoadResult,
    RawDatasetCandidate,
    discover_rawdata_files,
    load_rawdata_project,
)
from src.auto.runner import (
    AutoRawDataAnalysisResult,
    run_auto_rawdata_analysis,
)
from src.auto.validation import (
    AutoRunValidationItem,
    AutoRunValidationReport,
    validate_auto_run_outputs,
)
from src.auto.variable_inference import (
    AutoVariableInferenceResult,
    AutoVariableInferenceStep,
    VariableRoleInference,
    build_auto_variable_map,
    infer_variable_roles,
    role_inferences_to_dataframe,
    variable_map_to_dataframe,
)

__all__ = [
    "AutoAnalysisPlanDecision",
    "AutoAnalysisPlanResult",
    "AutoAnalysisPlanStep",
    "AutoRawDataAnalysisResult",
    "AutoRawDataLoadResult",
    "AutoRawDataLoadingStep",
    "AutoRegressionPipelineBuildResult",
    "AutoRunValidationItem",
    "AutoRunValidationReport",
    "AutoVariableInferenceResult",
    "AutoVariableInferenceStep",
    "AutoVariableRoleOverrides",
    "RawDatasetCandidate",
    "VariableRoleInference",
    "auto_analysis_plan_decisions_to_dataframe",
    "auto_analysis_plan_summary_to_dataframe",
    "apply_variable_role_overrides",
    "build_auto_analysis_plan",
    "build_auto_regression_orchestrator",
    "build_auto_variable_role_overrides",
    "build_auto_variable_map",
    "discover_rawdata_files",
    "infer_variable_roles",
    "load_rawdata_project",
    "register_auto_regression_pipeline",
    "role_inferences_to_dataframe",
    "run_auto_rawdata_analysis",
    "validate_auto_run_outputs",
    "variable_map_to_dataframe",
    "write_auto_analysis_config_files",
]
