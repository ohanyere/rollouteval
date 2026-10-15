"""Human-review records and deterministic Markdown packet rendering."""

from enum import StrEnum
import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rollouteval.evaluation.calibration import GroundednessCalibrationCase, GroundednessCalibrationSet
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
