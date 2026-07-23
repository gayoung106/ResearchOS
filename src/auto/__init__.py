"""Automatic project execution helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AutoAnalysisPlanDecision": "src.auto.analysis_plan",
    "AutoAnalysisPlanResult": "src.auto.analysis_plan",
    "AutoAnalysisPlanStep": "src.auto.analysis_plan",
    "auto_analysis_plan_decisions_to_dataframe": "src.auto.analysis_plan",
    "auto_analysis_plan_summary_to_dataframe": "src.auto.analysis_plan",
    "build_auto_analysis_plan": "src.auto.analysis_plan",
    "write_auto_analysis_config_files": "src.auto.analysis_plan",
    "AutoMultiOutcomeAnalysisPlanResult": "src.auto.multi_outcome",
    "AutoMultiOutcomeAnalysisPlanStep": "src.auto.multi_outcome",
    "AutoOutcomeAnalysisPlan": "src.auto.multi_outcome",
    "AutoOutcomeCandidate": "src.auto.multi_outcome",
    "auto_multi_outcome_candidates_to_dataframe": "src.auto.multi_outcome",
    "auto_multi_outcome_plans_to_dataframe": "src.auto.multi_outcome",
    "build_auto_multi_outcome_analysis_plans": "src.auto.multi_outcome",
    "infer_auto_outcome_candidates": "src.auto.multi_outcome",
    "AgentHypothesis": "src.auto.research_intent",
    "AgentResearchModel": "src.auto.research_intent",
    "AgentResearchModelValidationIssue": "src.auto.research_intent",
    "AgentResearchModelValidationReport": "src.auto.research_intent",
    "AgentVariableMatch": "src.auto.research_intent",
    "AutoResearchIntentAgentStep": "src.auto.research_intent",
    "ResearchConceptVariableMatch": "src.auto.research_intent",
    "ResearchIntent": "src.auto.research_intent",
    "ResearchIntentExtraction": "src.auto.research_intent",
    "ResearchIntentHypothesisCandidate": "src.auto.research_intent",
    "agent_research_model_from_dict": "src.auto.research_intent",
    "agent_research_model_to_dict": "src.auto.research_intent",
    "agent_research_model_validation_to_dataframe": "src.auto.research_intent",
    "apply_agent_research_model_to_analysis_plan": "src.auto.research_intent",
    "apply_agent_research_model_to_variable_map": "src.auto.research_intent",
    "build_claude_research_model_prompt": "src.auto.research_intent",
    "build_research_concept_variable_matches": "src.auto.research_intent",
    "build_research_context_packet": "src.auto.research_intent",
    "draft_agent_research_model_from_intent": "src.auto.research_intent",
    "infer_research_intent_structure": "src.auto.research_intent",
    "load_agent_research_model": "src.auto.research_intent",
    "load_research_intent": "src.auto.research_intent",
    "research_intent_to_dict": "src.auto.research_intent",
    "research_concept_variable_matches_to_dataframe": "src.auto.research_intent",
    "research_concept_variable_matches_to_dicts": "src.auto.research_intent",
    "research_intent_extraction_to_dict": "src.auto.research_intent",
    "validate_agent_research_model": "src.auto.research_intent",
    "write_agent_research_model": "src.auto.research_intent",
    "write_claude_research_model_prompt": "src.auto.research_intent",
    "write_research_context_packet": "src.auto.research_intent",
    "write_research_intent_template": "src.auto.research_intent",
    "AutoVariableRoleOverrides": "src.auto.overrides",
    "apply_variable_role_overrides": "src.auto.overrides",
    "build_auto_variable_role_overrides": "src.auto.overrides",
    "AutoMultiOutcomePipelineBuildResult": "src.auto.pipeline",
    "AutoMultiOutcomePipelineRunResult": "src.auto.pipeline",
    "AutoRegressionPipelineBuildResult": "src.auto.pipeline",
    "build_auto_multi_outcome_regression_orchestrators": "src.auto.pipeline",
    "run_auto_multi_outcome_regression_orchestrators": "src.auto.pipeline",
    "build_auto_regression_orchestrator": "src.auto.pipeline",
    "register_auto_regression_pipeline": "src.auto.pipeline",
    "AutoRawDataLoadResult": "src.auto.rawdata_loader",
    "AutoRawDataLoadingStep": "src.auto.rawdata_loader",
    "RawDatasetCandidate": "src.auto.rawdata_loader",
    "build_rawdata_quality_report": "src.auto.rawdata_loader",
    "discover_metadata_files": "src.auto.rawdata_loader",
    "discover_rawdata_files": "src.auto.rawdata_loader",
    "enrich_variable_metadata_from_files": "src.auto.rawdata_loader",
    "load_rawdata_project": "src.auto.rawdata_loader",
    "AutoRawDataAnalysisResult": "src.auto.runner",
    "run_auto_rawdata_analysis": "src.auto.runner",
    "AutoRunValidationItem": "src.auto.validation",
    "AutoRunValidationReport": "src.auto.validation",
    "auto_run_validation_report_to_dataframe": "src.auto.validation",
    "validate_auto_run_outputs": "src.auto.validation",
    "AutoVariableInferenceResult": "src.auto.variable_inference",
    "AutoVariableInferenceStep": "src.auto.variable_inference",
    "VariableRoleInference": "src.auto.variable_inference",
    "build_auto_variable_map": "src.auto.variable_inference",
    "infer_variable_roles": "src.auto.variable_inference",
    "role_inferences_to_dataframe": "src.auto.variable_inference",
    "variable_map_to_dataframe": "src.auto.variable_inference",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'src.auto' has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
