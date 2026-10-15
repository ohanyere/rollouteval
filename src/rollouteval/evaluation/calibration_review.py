"""Human-review records and deterministic Markdown packet rendering."""

from enum import StrEnum
import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pathlib import Path

import yaml
from pydantic import field_validator, model_validator

from rollouteval.evaluation.calibration import (
    GroundednessCalibrationCase,
    GroundednessCalibrationSet,
    load_groundedness_calibration,
)
from rollouteval.evaluation.results import GraderStatus


class GroundednessReviewDecision(StrEnum):
    """A human reviewer's disposition of a candidate groundedness label."""

    APPROVE = "APPROVE"
    CHANGE = "CHANGE"
    FLAG = "FLAG"


class HumanGroundednessReview(BaseModel):
    """One completed human review record; it is never judge input."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    case_id: str = Field(min_length=1)
    decision: GroundednessReviewDecision
    reviewed_status: GraderStatus | None
    reviewer_rationale: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    reviewed_at: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_completed_review(self) -> "HumanGroundednessReview":
        for field_name in ("case_id", "reviewer_rationale", "reviewer", "reviewed_at"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be blank")
        if (
            self.decision in (GroundednessReviewDecision.APPROVE, GroundednessReviewDecision.CHANGE)
            and self.reviewed_status is None
        ):
            raise ValueError(f"{self.decision.value} requires reviewed_status")
        return self

    def validate_against(self, candidate: GroundednessCalibrationCase) -> "HumanGroundednessReview":
        """Check case identity and the label-preservation rule for APPROVE."""
        if self.case_id != candidate.case_id:
            raise ValueError("review case_id does not match calibration candidate")
        if (
            self.decision is GroundednessReviewDecision.APPROVE
            and self.reviewed_status is not candidate.expected_status
        ):
            raise ValueError("APPROVE requires reviewed_status to equal candidate expected_status")
        return self


class HumanReviewedGroundednessReferenceSet(BaseModel):
    """Human-reviewed labels, separated from the evidence contexts."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    calibration_id: str = Field(min_length=1)
    reviews: tuple[HumanGroundednessReview, ...] = Field(min_length=1)

    @field_validator("calibration_id")
    @classmethod
    def require_nonblank_calibration_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("calibration_id must not be blank")
        return value

    @model_validator(mode="after")
    def validate_approved_unique_reviews(self) -> "HumanReviewedGroundednessReferenceSet":
        case_ids = [review.case_id for review in self.reviews]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("review case_id values must be unique")
        for review in self.reviews:
            if review.decision is not GroundednessReviewDecision.APPROVE:
                raise ValueError("human-reviewed reference set accepts only APPROVE reviews")
            if review.reviewed_status is None:
                raise ValueError("APPROVE review must include reviewed_status")
        return self

    def reviewed_status_for(self, case_id: str) -> GraderStatus:
        """Return the approved status for an exact case identifier."""
        for review in self.reviews:
            if review.case_id == case_id:
                assert review.reviewed_status is not None
                return review.reviewed_status
        raise KeyError(case_id)


class _HumanGroundednessReviewFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    calibration_id: str = Field(min_length=1)
    reviews: tuple[HumanGroundednessReview, ...] = Field(min_length=1)


def load_human_groundedness_reviews(
    path: Path,
    calibration_set: GroundednessCalibrationSet,
) -> tuple[HumanGroundednessReview, ...]:
    """Load a complete review file and validate each review against its case."""
    with path.open(encoding="utf-8") as review_file:
        data = yaml.safe_load(review_file)
    review_data = _HumanGroundednessReviewFile.model_validate(data)
    if review_data.calibration_id != calibration_set.calibration_id:
        raise ValueError("review calibration_id does not match calibration set")

    cases = {case.case_id: case for case in calibration_set.cases}
    reviews_by_id: dict[str, HumanGroundednessReview] = {}
    for review in review_data.reviews:
        if review.case_id in reviews_by_id:
            raise ValueError(f"duplicate review case_id: {review.case_id}")
        candidate = cases.get(review.case_id)
        if candidate is None:
            raise ValueError(f"unknown review case_id: {review.case_id}")
        review.validate_against(candidate)
        reviews_by_id[review.case_id] = review

    missing = [case.case_id for case in calibration_set.cases if case.case_id not in reviews_by_id]
    if missing:
        raise ValueError("missing reviews for case_id values: " + ", ".join(missing))
    return review_data.reviews


def load_human_reviewed_groundedness_reference(
    calibration_path: Path,
    reviews_path: Path,
) -> HumanReviewedGroundednessReferenceSet:
    """Load candidates and approved labels, ordering the reference by dataset."""
    calibration_set = load_groundedness_calibration(calibration_path)
    reviews = load_human_groundedness_reviews(reviews_path, calibration_set)
    reviews_by_id = {review.case_id: review for review in reviews}
    return HumanReviewedGroundednessReferenceSet(
        calibration_id=calibration_set.calibration_id,
        reviews=tuple(reviews_by_id[case.case_id] for case in calibration_set.cases),
    )


def render_groundedness_review(calibration_set: GroundednessCalibrationSet) -> str:
    """Render an unfilled, deterministic Markdown packet in source order."""
    sections = [
        "# Groundedness Calibration Human Review",
        "",
        f"Calibration set: `{calibration_set.calibration_id}`",
        "",
        "Review each candidate label against the supplied evidence. These are proposed labels, not approved ground truth.",
        "",
    ]
    for number, case in enumerate(calibration_set.cases, start=1):
        context = case.context
        sections.extend(
            [
                f"## {number}. {case.case_id}",
                "",
                f"**Description:** {case.description}",
                "",
                f"**Task:** {context.task}",
                "",
                f"**Declared decision:** {context.decision.value}",
                "",
                "**Final explanation:**",
                "",
                context.explanation,
                "",
                "**Observed evidence:**",
                "",
            ]
        )
        if context.evidence:
            for item in context.evidence:
                sections.extend(
                    [
                        f"- Interaction index: {item.interaction_index}",
                        f"  - Evidence ID: `{item.evidence_id.value}`",
                        f"  - Source tool: `{item.source_tool}`",
                        f"  - Status: `{item.status.value}`",
                        "  - Raw evidence content:",
                    ]
                )
                fence = _code_fence(item.content)
                sections.extend([f"{fence}text", item.content, fence, ""])
        else:
            sections.append("- No observation evidence was recorded.")
        if context.evidence:
            sections.pop()
        sections.extend(
            [
                "",
                f"**Proposed label:** `{case.expected_status.value}`",
                "",
                "**Candidate rationale:**",
                "",
                case.rationale,
                "",
                "**Human review:**",
                "- Decision: [ ] APPROVE  [ ] CHANGE  [ ] FLAG",
                "- Reviewed label:",
                "- Reviewer rationale:",
                "- Reviewer:",
                "- Review date:",
                "",
                "---",
                "",
            ]
        )
    return "\n".join(sections).rstrip() + "\n"


def _code_fence(content: str) -> str:
    longest = max((len(run) for run in re.findall(r"`+", content)), default=0)
    return "`" * max(3, longest + 1)
