"""Manual role overrides for automatic variable maps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.common.config_models import VariableMap

OverrideRole = Literal["dependent", "independent", "control", "cluster", "weight", "id", "time"]


@dataclass(slots=True)
class AutoVariableRoleOverrides:
    dependent_variable: str | None = None
    independent_variables: list[str] = field(default_factory=list)
    control_variables: list[str] = field(default_factory=list)
    cluster_variable: str | None = None
    weight_variable: str | None = None
    id_variable: str | None = None
    time_variable: str | None = None

    def has_overrides(self) -> bool:
        return any(
            [
                self.dependent_variable,
                self.independent_variables,
                self.control_variables,
                self.cluster_variable,
                self.weight_variable,
                self.id_variable,
                self.time_variable,
            ]
        )


def _normalize_list(values: list[str] | tuple[str, ...] | None) -> list[str]:
    return [str(value).strip() for value in values or [] if str(value).strip()]


def build_auto_variable_role_overrides(
    *,
    dependent_variable: str | None = None,
    independent_variables: list[str] | tuple[str, ...] | None = None,
    control_variables: list[str] | tuple[str, ...] | None = None,
    cluster_variable: str | None = None,
    weight_variable: str | None = None,
    id_variable: str | None = None,
    time_variable: str | None = None,
) -> AutoVariableRoleOverrides:
    return AutoVariableRoleOverrides(
        dependent_variable=dependent_variable.strip() if dependent_variable and dependent_variable.strip() else None,
        independent_variables=_normalize_list(independent_variables),
        control_variables=_normalize_list(control_variables),
        cluster_variable=cluster_variable.strip() if cluster_variable and cluster_variable.strip() else None,
        weight_variable=weight_variable.strip() if weight_variable and weight_variable.strip() else None,
        id_variable=id_variable.strip() if id_variable and id_variable.strip() else None,
        time_variable=time_variable.strip() if time_variable and time_variable.strip() else None,
    )


def _override_targets(overrides: AutoVariableRoleOverrides) -> dict[str, OverrideRole]:
    targets: dict[str, OverrideRole] = {}
    if overrides.dependent_variable:
        targets[overrides.dependent_variable] = "dependent"
    for variable in overrides.independent_variables:
        targets[variable] = "independent"
    for variable in overrides.control_variables:
        targets[variable] = "control"
    if overrides.cluster_variable:
        targets[overrides.cluster_variable] = "cluster"
    if overrides.weight_variable:
        targets[overrides.weight_variable] = "weight"
    if overrides.id_variable:
        targets[overrides.id_variable] = "id"
    if overrides.time_variable:
        targets[overrides.time_variable] = "time"
    return targets


def apply_variable_role_overrides(
    variable_map: VariableMap,
    overrides: AutoVariableRoleOverrides,
) -> VariableMap:
    if not overrides.has_overrides():
        return variable_map

    targets = _override_targets(overrides)
    missing = sorted(set(targets) - set(variable_map.variables))
    if missing:
        raise ValueError("Override variables are missing from the dataset: " + ", ".join(missing))

    output = variable_map.model_copy(deep=True)
    target_names = set(targets)
    for name, definition in output.variables.items():
        if name in targets:
            definition.role = targets[name]  # type: ignore[assignment]
            definition.review_status = "user_overridden"
            definition.evidence["role_override"] = True
            definition.evidence["role_override_role"] = targets[name]
            definition.notes = "Role was set by an automatic workflow override."
        elif definition.role in set(targets.values()) and definition.role in {
            "dependent",
            "cluster",
            "weight",
            "id",
            "time",
        }:
            definition.role = "independent" if name not in target_names else definition.role  # type: ignore[assignment]
            definition.evidence["role_override_displaced"] = True
    return output
