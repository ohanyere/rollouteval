"""Typed domain models for a rollout evaluation scenario."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Decision(StrEnum):
    PROMOTE = "promote"
    PAUSE = "pause"
    ROLLBACK = "rollback"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class RolloutState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    stable_version: str
    candidate_version: str
    candidate_weight: int = Field(ge=0, le=100)
    phase: str


class ServiceMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stable_error_rate: float = Field(ge=0)
    candidate_error_rate: float = Field(ge=0)
    stable_p95_ms: float = Field(ge=0)
    candidate_p95_ms: float = Field(ge=0)
    candidate_requests: int = Field(ge=0)


class RuntimeEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    severity: Severity
    message: str
    occurred_seconds_ago: int | None = Field(default=None, ge=0)


class InitialEnvironment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rollout: RolloutState
    metrics: ServiceMetrics | None = None
    events: tuple[RuntimeEvent, ...]


class ExpectedFinalState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    phase: str
    candidate_weight: int = Field(ge=0, le=100)


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    id: str
    name: str
    description: str
    task: str
    initial_environment: InitialEnvironment
    required_evidence: tuple[str, ...]
    expected_decision: Decision
    expected_final_state: ExpectedFinalState
    max_mutations: int = Field(default=1, ge=0)
