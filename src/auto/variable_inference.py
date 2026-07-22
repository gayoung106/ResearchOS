"""Automatic variable measurement-level and role inference."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config_models import VariableDefinition, VariableMap
from src.pipeline.context import ResearchContext
from src.pipeline.runtime import PipelineRuntime
from src.pipeline.step import PipelineStep, StepResult
from src.preprocess.detector import (
    VariableDetection,
    detect_dataframe_variables,
    detections_to_dataframe,
)

_ANALYZABLE_OUTCOMES = {"continuous", "binary", "ordinal", "count", "proportion", "scale_item"}
_PREDICTOR_LEVELS = _ANALYZABLE_OUTCOMES | {"nominal"}


@dataclass(slots=True)
class VariableRoleInference:
    variable_name: str
    role: str
    measurement_level: str
    confidence: float
    reason: str
    alternatives: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AutoVariableInferenceResult:
    variable_map: VariableMap
    detections: list[VariableDetection]
    role_inferences: list[VariableRoleInference]
    warnings: list[str] = field(default_factory=list)


def _normalize_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return normalized


def _tokens(name: str) -> set[str]:
    return {part for part in _normalize_name(name).split("_") if part}


def _contains_any(name: str, keywords: set[str]) -> bool:
    normalized = _normalize_name(name)
    tokens = _tokens(name)
    return bool(tokens & keywords) or any(keyword in normalized for keyword in keywords)


def _series_profile(dataframe: pd.DataFrame, variable_name: str) -> dict[str, Any]:
    series = dataframe[variable_name]
    non_missing = series.dropna()
    row_count = len(series)
    unique_count = int(non_missing.nunique())
    unique_rate = float(unique_count / max(len(non_missing), 1))
    return {
        "row_count": row_count,
        "non_missing_count": int(len(non_missing)),
        "unique_count": unique_count,
        "unique_rate": unique_rate,
        "is_numeric": bool(pd.api.types.is_numeric_dtype(series)),
        "is_datetime": bool(pd.api.types.is_datetime64_any_dtype(series)),
    }


def _special_role_by_name(
    variable_name: str,
    detection: VariableDetection,
    profile: dict[str, Any],
) -> VariableRoleInference | None:
    name = _normalize_name(variable_name)
    tokens = _tokens(variable_name)
    level = detection.detected_level

    if profile["is_datetime"] or _contains_any(name, {"date", "datetime", "time", "year", "month", "wave", "period"}):
        return VariableRoleInference(variable_name, "time", level, 0.9, "Name or dtype suggests a time variable.")

    if _contains_any(name, {"weight", "weights", "wt", "wgt", "sampling_weight"}):
        return VariableRoleInference(variable_name, "weight", level, 0.9, "Name suggests an analytic weight variable.")

    if _contains_any(name, {"strata", "stratum"}):
        return VariableRoleInference(variable_name, "strata", level, 0.9, "Name suggests a strata variable.")

    if _contains_any(name, {"cluster", "group", "site", "school", "class", "clinic", "hospital", "center", "team"}):
        confidence = 0.85 if profile["unique_count"] >= 3 else 0.65
        return VariableRoleInference(variable_name, "cluster", level, confidence, "Name suggests a clustering or grouping variable.")

    if name in {"id", "caseid", "case_id"} or name.endswith("_id") or "id" in tokens:
        confidence = 0.95 if profile["unique_rate"] >= 0.8 else 0.75
        return VariableRoleInference(variable_name, "id", level, confidence, "Name suggests an identifier variable.")

    return None


def _dependent_score(variable_name: str, detection: VariableDetection, profile: dict[str, Any], index: int) -> tuple[float, str]:
    name = _normalize_name(variable_name)
    tokens = _tokens(variable_name)
    level = detection.detected_level
    if level not in _ANALYZABLE_OUTCOMES:
        return (float("-inf"), "Variable is not an analyzable outcome level.")
    if profile["unique_count"] <= 1:
        return (float("-inf"), "Variable is constant.")

    score = 0.0
    reason = "First analyzable variable was selected as a provisional dependent variable."
    exact_names = {"y", "dv", "outcome", "target", "response", "result", "score", "total"}
    if name in exact_names or tokens & exact_names:
        score += 100.0
        reason = "Name strongly suggests an outcome variable."
    if _contains_any(name, {"outcome", "result", "score", "total", "dependent", "response", "post", "followup"}):
        score += 50.0
        reason = "Name suggests an outcome variable."
    if level in {"continuous", "count", "proportion"}:
        score += 20.0
    elif level in {"binary", "ordinal", "scale_item"}:
        score += 10.0
    score += max(0.0, 20.0 - index)
    return (score, reason)


def infer_variable_roles(
    dataframe: pd.DataFrame,
    detections: list[VariableDetection],
) -> list[VariableRoleInference]:
    detection_map = {detection.variable_name: detection for detection in detections}
    special_roles: dict[str, VariableRoleInference] = {}
    dependent_candidates: list[tuple[float, int, str, str]] = []

    for index, variable_name in enumerate(dataframe.columns):
        detection = detection_map[str(variable_name)]
        profile = _series_profile(dataframe, str(variable_name))
        special = _special_role_by_name(str(variable_name), detection, profile)
        if special is not None:
            special_roles[str(variable_name)] = special
            continue
        score, reason = _dependent_score(str(variable_name), detection, profile, index)
        if score != float("-inf"):
            dependent_candidates.append((score, -index, str(variable_name), reason))

    dependent_name: str | None = None
    dependent_reason = ""
    if dependent_candidates:
        _, _, dependent_name, dependent_reason = max(dependent_candidates)

    output: list[VariableRoleInference] = []
    for variable_name in [str(column) for column in dataframe.columns]:
        detection = detection_map[variable_name]
        profile = _series_profile(dataframe, variable_name)
        if variable_name in special_roles:
            output.append(special_roles[variable_name])
            continue
        if variable_name == dependent_name:
            output.append(
                VariableRoleInference(
                    variable_name,
                    "dependent",
                    detection.detected_level,
                    0.85 if "strongly" in dependent_reason or "suggests" in dependent_reason else 0.55,
                    dependent_reason,
                    alternatives=["independent", "control"],
                )
            )
            continue
        if detection.detected_level in _PREDICTOR_LEVELS and profile["unique_count"] > 1:
            output.append(
                VariableRoleInference(
                    variable_name,
                    "independent",
                    detection.detected_level,
                    0.65,
                    "Variable has analyzable variation and is not reserved for a special role.",
                    alternatives=["control", "other"],
                )
            )
        else:
            output.append(
                VariableRoleInference(
                    variable_name,
                    "other",
                    detection.detected_level,
                    0.5,
                    "Variable is not suitable for automatic regression roles without review.",
                    alternatives=["control"],
                )
            )
    return output


def build_auto_variable_map(
    dataframe: pd.DataFrame,
    *,
    variable_metadata: pd.DataFrame | None = None,
) -> AutoVariableInferenceResult:
    detections = detect_dataframe_variables(dataframe, variable_metadata=variable_metadata)
    role_inferences = infer_variable_roles(dataframe, detections)
    warnings = [
        f"{item.variable_name}: {item.role} confidence={item.confidence:.2f}"
        for item in role_inferences
        if item.confidence < 0.7
    ]
    variable_map = VariableMap(
        variables={
            item.variable_name: VariableDefinition(
                original_name=item.variable_name,
                role=item.role,  # type: ignore[arg-type]
                measurement_level=item.measurement_level,  # type: ignore[arg-type]
                evidence={
                    "auto_role_confidence": item.confidence,
                    "auto_role_reason": item.reason,
                    "auto_role_alternatives": item.alternatives,
                },
                review_status="auto_inferred",
                notes=item.reason,
            )
            for item in role_inferences
        }
    )
    return AutoVariableInferenceResult(
        variable_map=variable_map,
        detections=detections,
        role_inferences=role_inferences,
        warnings=warnings,
    )


def role_inferences_to_dataframe(role_inferences: list[VariableRoleInference]) -> pd.DataFrame:
    rows = []
    for item in role_inferences:
        row = asdict(item)
        row["alternatives"] = " | ".join(item.alternatives)
        rows.append(row)
    return pd.DataFrame(rows)


def variable_map_to_dataframe(variable_map: VariableMap) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variable_name": name,
                "role": definition.role,
                "measurement_level": definition.measurement_level,
                "review_status": definition.review_status,
                "confidence": definition.evidence.get("auto_role_confidence"),
                "reason": definition.evidence.get("auto_role_reason"),
                "alternatives": " | ".join(definition.evidence.get("auto_role_alternatives", [])),
            }
            for name, definition in variable_map.variables.items()
        ]
    )


class AutoVariableInferenceStep(PipelineStep):
    """Infer measurement levels and provisional analysis roles for loaded rawdata."""

    def __init__(self, runtime: PipelineRuntime, *, order: int = 20) -> None:
        super().__init__(name="02_auto_variable_inference", order=order, required=True)
        self.runtime = runtime

    def run(self, context: ResearchContext, working_directory: Path) -> StepResult:
        dataframe = self.runtime.require_dataframe()
        result = build_auto_variable_map(
            dataframe,
            variable_metadata=self.runtime.variable_metadata,
        )
        self.runtime.detections = result.detections
        self.runtime.set_artifact("auto_variable_inference_result", result)
        self.runtime.set_artifact("auto_variable_map", result.variable_map)

        output_dir = working_directory / "result" / "02_auto_variables"
        output_dir.mkdir(parents=True, exist_ok=True)
        detections_path = output_dir / "variable_detections.xlsx"
        roles_path = output_dir / "variable_role_inference.xlsx"
        map_path = output_dir / "inferred_variable_map.xlsx"

        detections_to_dataframe(result.detections).to_excel(detections_path, index=False)
        role_inferences_to_dataframe(result.role_inferences).to_excel(roles_path, index=False)
        variable_map_to_dataframe(result.variable_map).to_excel(map_path, index=False)

        return StepResult(
            stage_name=self.name,
            success=True,
            output_files=[str(detections_path), str(roles_path), str(map_path)],
            warnings=result.warnings,
            metadata={
                "variable_count": len(result.role_inferences),
                "dependent_count": sum(item.role == "dependent" for item in result.role_inferences),
                "independent_count": sum(item.role == "independent" for item in result.role_inferences),
                "review_warning_count": len(result.warnings),
            },
        )
