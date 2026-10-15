"""Reviewable groundedness calibration cases and an offline scoring runner."""

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rollouteval.evaluation.groundedness import GroundednessJudge
from rollouteval.evaluation.grounding import GroundingContext
from rollouteval.evaluation.results import GraderStatus


NonblankText = Annotated[str, Field(min_length=1)]


class GroundednessCalibrationCase(BaseModel):
    """One hand-authored, reviewable context and its expected label."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    case_id: NonblankText
    description: NonblankText
    context: GroundingContext
    expected_status: GraderStatus
    rationale: NonblankText

    @field_validator("case_id", "description", "rationale")
    @classmethod
    def require_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class GroundednessCalibrationSet(BaseModel):
    """Ordered calibration candidates with unique identifiers."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    calibration_id: NonblankText
    cases: tuple[GroundednessCalibrationCase, ...] = Field(min_length=1)

    @field_validator("calibration_id")
    @classmethod
    def require_nonblank_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("calibration_id must not be blank")
        return value

    @model_validator(mode="after")
    def require_unique_case_ids(self) -> "GroundednessCalibrationSet":
        identifiers = [case.case_id for case in self.cases]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("case_id values must be unique")
        return self


class GroundednessCalibrationResult(BaseModel):
    """One label comparison; matched is derived from the two statuses."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    case_id: NonblankText
    expected_status: GraderStatus
    actual_status: GraderStatus
    judge_reason: NonblankText

    @field_validator("case_id", "judge_reason")
    @classmethod
    def require_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value

    @property
    def matched(self) -> bool:
        return self.expected_status is self.actual_status


class GroundednessStatusAgreement(BaseModel):
    """Exact label counts for one reference status."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: GraderStatus
    total: int = Field(ge=0)
    matched: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_matched_bound(self) -> "GroundednessStatusAgreement":
        if self.matched > self.total:
            raise ValueError("matched cannot exceed total")
        return self


class GroundednessCalibrationReport(BaseModel):
    """Ordered exact-label agreement results without a quality threshold."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    calibration_id: NonblankText
    results: tuple[GroundednessCalibrationResult, ...] = Field(min_length=1)

    @field_validator("calibration_id")
    @classmethod
    def require_nonblank_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("calibration_id must not be blank")
        return value

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def matched_cases(self) -> int:
        return sum(result.matched for result in self.results)

    @property
    def mismatched_cases(self) -> int:
        return self.total_cases - self.matched_cases

    @property
    def exact_match_rate(self) -> float:
        """Fraction of cases whose status exactly matched the expected status."""
        return self.matched_cases / self.total_cases

    @property
    def mismatches(self) -> tuple[GroundednessCalibrationResult, ...]:
        """Return mismatched cases in calibration order."""
        return tuple(result for result in self.results if not result.matched)

    @property
    def status_agreement(self) -> tuple[GroundednessStatusAgreement, ...]:
        """Return total and matched counts grouped by reference status."""
        return tuple(
            GroundednessStatusAgreement(
                status=status,
                total=sum(result.expected_status is status for result in self.results),
                matched=sum(
                    result.expected_status is status and result.matched
                    for result in self.results
                ),
            )
            for status in GraderStatus
        )


def load_groundedness_calibration(path: Path) -> GroundednessCalibrationSet:
    """Load and strictly validate one YAML groundedness calibration file."""
    with path.open(encoding="utf-8") as dataset_file:
        data = yaml.safe_load(dataset_file)
    return GroundednessCalibrationSet.model_validate(data)


class GroundednessCalibrationRunner:
    """Compare a judge's status labels against ordered expected labels."""

    def __init__(self, judge: GroundednessJudge) -> None:
        self._judge = judge

    def run(
        self,
        calibration_set: GroundednessCalibrationSet,
    ) -> GroundednessCalibrationReport:
        """Call the judge once per context and preserve case order."""
        results = tuple(
            GroundednessCalibrationResult(
                case_id=case.case_id,
                expected_status=case.expected_status,
                actual_status=judgment.status,
                judge_reason=judgment.reason,
            )
            for case in calibration_set.cases
            for judgment in (self._judge.judge(case.context),)
        )
        return GroundednessCalibrationReport(
            calibration_id=calibration_set.calibration_id,
            results=results,
        )
