#!/usr/bin/env python3
"""Run live groundedness calibration against the human-reviewed reference."""

import argparse
from pathlib import Path
from typing import Sequence

from rollouteval.evaluation.calibration import (
    GroundednessCalibrationReport,
    load_groundedness_calibration,
)
from rollouteval.evaluation.calibration_review import (
    HumanReviewedGroundednessCalibrationRunner,
    load_human_reviewed_groundedness_reference,
)
from rollouteval.evaluation.providers import (
    GroqGroundednessJudge,
    OpenAIGroundednessJudge,
)
from rollouteval.provider_client import configured_api_provider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("groq", "openai"), required=True)
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier used for groundedness judging",
    )
    return parser


def format_groundedness_calibration_report(
    report: GroundednessCalibrationReport,
    *,
    model: str,
) -> str:
    """Format exact label agreement and ordered per-case judgments."""
    lines = [
        f"Calibration ID: {report.calibration_id}",
        f"Judge model: {model}",
        f"Total cases: {report.total_cases}",
        f"Matched: {report.matched_cases}",
        f"Mismatched: {report.mismatched_cases}",
        f"Exact agreement: {report.exact_match_rate:.6f}",
        "",
    ]
    for result in report.results:
        match_label = "MATCH" if result.matched else "MISMATCH"
        lines.extend(
            [
                result.case_id,
                f"Reference: {result.expected_status.value}",
                f"Judge: {result.actual_status.value}",
                f"Result: {match_label}",
                f"Reason: {result.judge_reason}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if configured_api_provider(args.provider) is None:
        key_name = f"{args.provider.upper()}_API_KEY"
        raise SystemExit(f"{key_name} must be configured")
    project_root = Path(__file__).resolve().parents[1]
    calibration_path = project_root / "calibration" / "groundedness_v1.yaml"
    reviews_path = project_root / "calibration" / "groundedness_v1_reviews.yaml"
    calibration_set = load_groundedness_calibration(calibration_path)
    reference_set = load_human_reviewed_groundedness_reference(
        calibration_path,
        reviews_path,
    )
    judge_type = (
        GroqGroundednessJudge
        if args.provider == "groq"
        else OpenAIGroundednessJudge
    )
    judge = judge_type(model=args.model)
    report = HumanReviewedGroundednessCalibrationRunner(judge).run(
        calibration_set,
        reference_set,
    )
    print(format_groundedness_calibration_report(report, model=args.model), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
