"""Evidence-bounded context construction for a future groundedness judge."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rollouteval.agent.contracts import AgentRunResult, AgentToolName
from rollouteval.domain.scenario import Decision, Scenario
from rollouteval.evaluation.evidence import (
    EvidenceExtractionError,
    EvidenceId,
    EvidenceSourceTool,
    EvidenceStatus,
    extract_evidence,
)

_TOOL_TO_EVIDENCE = {
    AgentToolName.GET_ROLLOUT_STATE: EvidenceId.ROLLOUT_STATE,
    AgentToolName.GET_SERVICE_METRICS: EvidenceId.SERVICE_METRICS,
    AgentToolName.GET_RECENT_EVENTS: EvidenceId.RUNTIME_EVENTS,
}


class GroundingEvidenceItem(BaseModel):
    """One observation-tool result or error the agent actually received."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    interaction_index: int = Field(ge=0)
    evidence_id: EvidenceId
    source_tool: EvidenceSourceTool
    status: EvidenceStatus
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def validate_nonblank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value

    @model_validator(mode="after")
    def validate_evidence_tool_pair(self) -> "GroundingEvidenceItem":
        expected = _TOOL_TO_EVIDENCE[AgentToolName(self.source_tool)]
        if self.evidence_id is not expected:
            raise ValueError("evidence_id does not match source_tool")
        return self


class GroundingContext(BaseModel):
    """Safe task, final declared answer, and only evidence seen by the agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task: str = Field(min_length=1)
    decision: Decision
    explanation: str = Field(min_length=1)
    evidence: tuple[GroundingEvidenceItem, ...]

    @field_validator("task", "explanation")
    @classmethod
    def validate_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


def build_grounding_context(
    scenario: Scenario,
    run_result: AgentRunResult,
) -> GroundingContext:
    """Copy only task, final answer, and chronologically observed evidence."""
    extracted = extract_evidence(run_result)
    evidence_items: list[GroundingEvidenceItem] = []
    for observation in extracted:
        interaction = run_result.interactions[observation.interaction_index]
        content = interaction.error if observation.status is EvidenceStatus.ERROR else interaction.result
        if content is None:
            raise EvidenceExtractionError(
                "observation evidence has neither result nor semantic error "
                f"at interaction index {observation.interaction_index}"
            )
        evidence_items.append(
            GroundingEvidenceItem(
                interaction_index=observation.interaction_index,
                evidence_id=observation.evidence_id,
                source_tool=observation.source_tool,
                status=observation.status,
                content=content,
            )
        )

    return GroundingContext(
        task=scenario.task,
        decision=run_result.final_answer.decision,
        explanation=run_result.final_answer.explanation,
        evidence=tuple(evidence_items),
    )
