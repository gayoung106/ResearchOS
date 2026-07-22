"""Automatic project execution helpers."""

from src.auto.rawdata_loader import (
    AutoRawDataLoadingStep,
    AutoRawDataLoadResult,
    RawDatasetCandidate,
    discover_rawdata_files,
    load_rawdata_project,
)

__all__ = [
    "AutoRawDataLoadResult",
    "RawDatasetCandidate",
    "AutoRawDataLoadingStep",
    "discover_rawdata_files",
    "load_rawdata_project",
    "AutoVariableInferenceResult",
    "AutoVariableInferenceStep",
    "VariableRoleInference",
    "build_auto_variable_map",
    "infer_variable_roles",
    "role_inferences_to_dataframe",
    "variable_map_to_dataframe",
]

from src.auto.variable_inference import (
    AutoVariableInferenceResult,
    AutoVariableInferenceStep,
    VariableRoleInference,
    build_auto_variable_map,
    infer_variable_roles,
    role_inferences_to_dataframe,
    variable_map_to_dataframe,
)
