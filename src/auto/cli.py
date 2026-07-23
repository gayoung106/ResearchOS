"""Command-line entry point for the automatic rawdata workflow."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from src.auto.runner import run_auto_rawdata_analysis


def _find_output_file(output_files: list[str], filename: str) -> str | None:
    for output_file in output_files:
        if Path(output_file).name == filename:
            return output_file
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.auto.cli",
        description="Run the automatic rawdata analysis workflow.",
    )
    parser.add_argument(
        "--working-directory",
        default=".",
        help="Project directory. Defaults to the current directory.",
    )
    parser.add_argument(
        "--rawdata-dir",
        default="rawdata",
        help="Directory containing raw data files, relative to the working directory unless absolute.",
    )
    parser.add_argument(
        "--source-file",
        default=None,
        help="Optional explicit data file to analyze instead of scanning rawdata-dir.",
    )
    parser.add_argument(
        "--no-auto-merge",
        action="store_true",
        help="Disable conservative ID-based merging across multiple rawdata files.",
    )
    parser.add_argument(
        "--codebook-dir",
        default="codebook",
        help="Directory containing codebook files used to enrich variable labels and roles.",
    )
    parser.add_argument(
        "--questionnaire-dir",
        default="questionnaire",
        help="Directory containing questionnaire files used to enrich question text and labels.",
    )
    parser.add_argument(
        "--project-name",
        default="auto_rawdata_analysis",
        help="Project name stored in the ResearchContext.",
    )
    parser.add_argument(
        "--model-id",
        default="main_model",
        help="Model id used for regression artifacts.",
    )
    parser.add_argument(
        "--enable-robustness",
        action="store_true",
        help="Enable registered robustness checks after automatic planning.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Load rawdata, infer variables, build the plan, and register the pipeline without running models.",
    )
    parser.add_argument(
        "--multi-outcome",
        action="store_true",
        help="Build and run one registered analysis pipeline per inferred outcome candidate.",
    )
    parser.add_argument(
        "--max-outcomes",
        type=int,
        default=3,
        help="Maximum number of inferred outcomes to analyze when --multi-outcome is enabled.",
    )
    parser.add_argument(
        "--dependent-variable",
        default=None,
        help="Override the inferred dependent variable.",
    )
    parser.add_argument(
        "--independent-variables",
        nargs="*",
        default=None,
        help="Override inferred independent variables. Example: --independent-variables x1 x2",
    )
    parser.add_argument(
        "--control-variables",
        nargs="*",
        default=None,
        help="Override inferred control variables.",
    )
    parser.add_argument(
        "--cluster-variable",
        default=None,
        help="Override the inferred cluster/group variable.",
    )
    parser.add_argument(
        "--weight-variable",
        default=None,
        help="Override the inferred weight variable.",
    )
    parser.add_argument(
        "--id-variable",
        default=None,
        help="Override the inferred entity/id variable.",
    )
    parser.add_argument(
        "--time-variable",
        default=None,
        help="Override the inferred time variable.",
    )
    parser.add_argument(
        "--research-intent-file",
        default=None,
        help="YAML or text file describing the research intent for Claude agent context generation.",
    )
    parser.add_argument(
        "--research-intent-text",
        default=None,
        help="Inline research intent text for Claude agent context generation.",
    )
    parser.add_argument(
        "--agent-research-model-file",
        default=None,
        help="YAML file returned by Claude to validate and apply before pipeline registration.",
    )
    parser.add_argument(
        "--no-apply-agent-model",
        action="store_true",
        help="Validate the Claude model YAML but do not apply it to the generated analysis plan.",
    )
    parser.add_argument(
        "--apply-draft-agent-model",
        action="store_true",
        help="Apply ResearchOS's conservative draft agent model when no Claude model YAML is available.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_auto_rawdata_analysis(
        Path(args.working_directory),
        rawdata_dir=args.rawdata_dir,
        source_file=args.source_file,
        auto_merge=not args.no_auto_merge,
        codebook_dir=args.codebook_dir,
        questionnaire_dir=args.questionnaire_dir,
        project_name=args.project_name,
        enable_robustness=args.enable_robustness,
        run_analysis=not args.plan_only,
        model_id=args.model_id,
        dependent_variable=args.dependent_variable,
        independent_variables=args.independent_variables,
        control_variables=args.control_variables,
        cluster_variable=args.cluster_variable,
        weight_variable=args.weight_variable,
        id_variable=args.id_variable,
        time_variable=args.time_variable,
        enable_multi_outcome=args.multi_outcome,
        max_outcomes=args.max_outcomes,
        research_intent_file=args.research_intent_file,
        research_intent_text=args.research_intent_text,
        agent_research_model_file=args.agent_research_model_file,
        apply_agent_model=not args.no_apply_agent_model,
        apply_draft_model=args.apply_draft_agent_model,
    )

    status = "completed" if result.success else "failed"
    print(f"Auto rawdata analysis {status}.")
    if result.failed_stage:
        print(f"Failed stage: {result.failed_stage}")
    if result.pipeline_build_result and result.pipeline_build_result.registration:
        registration = result.pipeline_build_result.registration
        print(f"Model type: {registration.model_type}")
        print(f"Dependent variable: {registration.dependent_variable}")
        print(f"Independent variables: {', '.join(registration.independent_variables)}")
    multi_build_result = getattr(result, "multi_outcome_pipeline_build_result", None)
    if multi_build_result is not None:
        print(f"Multi-outcome models: {len(multi_build_result.model_results)}")
    if result.output_files:
        final_report = _find_output_file(result.output_files, "auto_final_report.md")
        manifest = _find_output_file(result.output_files, "output_manifest.xlsx")
        recovery = _find_output_file(result.output_files, "auto_recovery_guide.xlsx")
        if final_report:
            print(f"Final report: {final_report}")
        if manifest:
            print(f"Output manifest: {manifest}")
        if recovery:
            print(f"Recovery guide: {recovery}")
        print("Output files:")
        for output_file in result.output_files:
            print(f"- {output_file}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
