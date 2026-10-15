"""Model-facing, evaluator-truth-free interaction contracts."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class ModelUsage(BaseModel):
    """Provider-reported token counts for one model invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    cache_write_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)


class AgentModelResponse(BaseModel):
    """Semantic model output with optional operational invocation metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    output: AgentOutput
    model: str | None = Field(default=None, min_length=1)
    usage: ModelUsage | None = None


class AgentToolInteraction(BaseModel):
    """One completed call and its JSON-encoded safe result or expected error."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call: AgentToolCall
    result: str | None = None
    error: str | None = Field(default=None, min_length=1)

    @classmethod
    def successful(cls, call: AgentToolCall, result: str) -> "AgentToolInteraction":
        return cls(call=call, result=result)

    @classmethod
    def failed(cls, call: AgentToolCall, error: str) -> "AgentToolInteraction":
        return cls(call=call, error=error)

    @model_validator(mode="after")
    def validate_outcome(self) -> "AgentToolInteraction":
        if (self.result is None) == (self.error is None):
            raise ValueError("exactly one of result or error must be provided")
        return self


class AgentRunResult(BaseModel):
    """Final semantic outcome of one bounded agent run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    final_answer: AgentFinalAnswer
    interactions: tuple[AgentToolInteraction, ...]
    steps: int = Field(ge=1)
