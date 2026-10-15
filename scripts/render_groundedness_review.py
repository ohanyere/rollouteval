#!/usr/bin/env python3
"""Generate the deterministic human-review packet for groundedness candidates."""

from pathlib import Path

from rollouteval.evaluation.calibration import load_groundedness_calibration
from rollouteval.evaluation.calibration_review import render_groundedness_review


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "calibration" / "groundedness_v1.yaml"
    output_path = project_root / "calibration" / "groundedness_v1_review.md"
    calibration_set = load_groundedness_calibration(dataset_path)
    output_path.write_text(render_groundedness_review(calibration_set), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
