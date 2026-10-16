"""Shared immutable result contract for deterministic graders."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class GraderStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


class GraderResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    grader: str = Field(min_length=1)
    status: GraderStatus
    reason: str = Field(min_length=1)
    evaluator_version: str = "1"
    criterion_id: str | None = None
    evidence_references: tuple[str, ...] = ()
    judge_provider: str | None = None
    judge_model: str | None = None
    prompt_sha256: str | None = None
    judge_error: str | None = None
