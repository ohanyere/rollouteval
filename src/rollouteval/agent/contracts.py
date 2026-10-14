"""Model-facing, evaluator-truth-free interaction contracts."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from rollouteval.domain.scenario import Decision, Scenario


class AgentToolName(StrEnum):
    """The complete set of tools currently available to an agent."""

    GET_ROLLOUT_STATE = "get_rollout_state"
    GET_SERVICE_METRICS = "get_service_metrics"
    GET_RECENT_EVENTS = "get_recent_events"
    PROMOTE_ROLLOUT = "promote_rollout"
    PAUSE_ROLLOUT = "pause_rollout"
    ROLLBACK_ROLLOUT = "rollback_rollout"


class AgentTask(BaseModel):
    """Safe task text for a future model request; contains no Scenario link."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    task: str = Field(min_length=1)

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "AgentTask":
        """Copy only the agent-visible task text from a validated scenario."""
        return cls(task=scenario.task)


class _NoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AgentToolCall(BaseModel):
    """A request to call one explicitly allowed no-argument tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["tool_call"]
    tool: AgentToolName
    arguments: _NoArguments = Field(default_factory=_NoArguments)


class AgentFinalAnswer(BaseModel):
    """An agent's externally visible decision and explanation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    kind: Literal["final"]
    decision: Decision
    explanation: str = Field(min_length=1)


AgentOutput = Annotated[AgentToolCall | AgentFinalAnswer, Field(discriminator="kind")]
