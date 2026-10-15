"""Evidence extraction and sufficiency grading from semantic run history."""

import json
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rollouteval.agent.contracts import AgentRunResult, AgentToolName
from rollouteval.domain.scenario import Scenario
from rollouteval.evaluation.results import GraderResult, GraderStatus


class EvidenceId(StrEnum):
    ROLLOUT_STATE = "rollout_state"
    SERVICE_METRICS = "service_metrics"
    RUNTIME_EVENTS = "runtime_events"


class EvidenceStatus(StrEnum):
    OBSERVED = "OBSERVED"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"


EvidenceSourceTool = Literal[
    "get_rollout_state",
    "get_service_metrics",
    "get_recent_events",
]


class EvidenceObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: EvidenceId
    source_tool: EvidenceSourceTool
    interaction_index: int = Field(ge=0)
    status: EvidenceStatus


class EvidenceExtractionError(ValueError):
    """Raised when a successful observation has malformed internal result data."""


_OBSERVATION_TO_EVIDENCE = {
    AgentToolName.GET_ROLLOUT_STATE: EvidenceId.ROLLOUT_STATE,
    AgentToolName.GET_SERVICE_METRICS: EvidenceId.SERVICE_METRICS,
    AgentToolName.GET_RECENT_EVENTS: EvidenceId.RUNTIME_EVENTS,
}


def extract_evidence(run_result: AgentRunResult) -> tuple[EvidenceObservation, ...]:
    """Extract every observation-tool outcome in its original interaction order."""
    observations: list[EvidenceObservation] = []
    for index, interaction in enumerate(run_result.interactions):
        tool = interaction.call.tool
        evidence_id = _OBSERVATION_TO_EVIDENCE.get(tool)
        if evidence_id is None:
            continue

        if interaction.error is not None:
            status = EvidenceStatus.ERROR
        elif evidence_id is EvidenceId.SERVICE_METRICS:
            status = _service_metrics_status(interaction.result, index)
        else:
            status = EvidenceStatus.OBSERVED

        observations.append(
            EvidenceObservation(
                evidence_id=evidence_id,
                source_tool=tool.value,
                interaction_index=index,
                status=status,
            )
        )
    return tuple(observations)


def _service_metrics_status(result: str | None, interaction_index: int) -> EvidenceStatus:
    try:
        payload = json.loads(result) if result is not None else None
    except json.JSONDecodeError as exc:
        raise EvidenceExtractionError(
            f"malformed service_metrics JSON at interaction index {interaction_index}"
        ) from exc

    if not isinstance(payload, dict) or type(payload.get("available")) is not bool:
        raise EvidenceExtractionError(
            "service_metrics result must contain a boolean 'available' field "
            f"at interaction index {interaction_index}"
        )

    if payload["available"]:
        if not isinstance(payload.get("metrics"), dict):
            raise EvidenceExtractionError(
                "available service_metrics result must contain a metrics object "
                f"at interaction index {interaction_index}"
            )
        return EvidenceStatus.OBSERVED

    if payload.get("metrics") is not None:
        raise EvidenceExtractionError(
            "unavailable service_metrics result must have null metrics "
            f"at interaction index {interaction_index}"
        )
    if not isinstance(payload.get("message"), str) or not payload["message"].strip():
        raise EvidenceExtractionError(
            "unavailable service_metrics result must include a message "
            f"at interaction index {interaction_index}"
        )
    return EvidenceStatus.UNAVAILABLE


class EvidenceSufficiencyGrader:
    """Grade whether every required evidence source was successfully observed."""

    def grade(self, scenario: Scenario, run_result: AgentRunResult) -> GraderResult:
        extracted = extract_evidence(run_result)
        evidence_by_id: dict[str, list[EvidenceStatus]] = {}
        for observation in extracted:
            evidence_by_id.setdefault(observation.evidence_id.value, []).append(
                observation.status
            )

        required = tuple(dict.fromkeys(scenario.required_evidence))
        missing = [identifier for identifier in required if identifier not in evidence_by_id]
        unavailable_or_error: list[str] = []
        for identifier in required:
            statuses = evidence_by_id.get(identifier, [])
            if statuses and EvidenceStatus.OBSERVED not in statuses:
                states = ", ".join(
                    status.value.lower()
                    for status in EvidenceStatus
                    if status in statuses
                )
                unavailable_or_error.append(f"{identifier} ({states})")

        if missing:
            return GraderResult(
                grader="evidence_sufficiency",
                status=GraderStatus.FAIL,
                reason="Required evidence never requested: " + ", ".join(missing) + ".",
            )
        if unavailable_or_error:
            return GraderResult(
                grader="evidence_sufficiency",
                status=GraderStatus.INCONCLUSIVE,
                reason=(
                    "Required evidence was attempted but unavailable or errored: "
                    + ", ".join(unavailable_or_error)
                    + "."
                ),
            )

        return GraderResult(
            grader="evidence_sufficiency",
            status=GraderStatus.PASS,
            reason="All required evidence was observed: " + ", ".join(required) + ".",
        )
