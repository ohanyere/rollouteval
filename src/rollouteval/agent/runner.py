"""Synchronous, bounded execution of model-requested rollout tools."""

from json import dumps
from typing import Protocol

from pydantic import BaseModel

from rollouteval.agent.contracts import (
    AgentFinalAnswer,
    AgentOutput,
    AgentRunResult,
    AgentTask,
    AgentToolCall,
    AgentToolInteraction,
    AgentToolName,
)
from rollouteval.environment import MutationLimitExceeded
from rollouteval.tools import RolloutMutationTools, RolloutObservationTools


class AgentModel(Protocol):
    """Provider-neutral interface for one synchronous model response."""

    def respond(
        self,
        task: AgentTask,
        interactions: tuple[AgentToolInteraction, ...],
    ) -> AgentOutput:
        """Return the next tool call or final answer."""
        ...


class AgentStepLimitExceeded(RuntimeError):
    """Raised when no final answer is produced within the configured steps."""


class RolloutToolDispatcher:
    """Explicitly dispatch the six supported calls to concrete rollout tools."""

    def __init__(
        self,
        observation_tools: RolloutObservationTools,
        mutation_tools: RolloutMutationTools,
    ) -> None:
        self._observation_tools = observation_tools
        self._mutation_tools = mutation_tools

    def execute(self, call: AgentToolCall) -> AgentToolInteraction:
        """Execute exactly the requested capability and encode its safe result."""
        try:
            match call.tool:
                case AgentToolName.GET_ROLLOUT_STATE:
                    result = self._observation_tools.get_rollout_state()
                case AgentToolName.GET_SERVICE_METRICS:
                    result = self._observation_tools.get_service_metrics()
                case AgentToolName.GET_RECENT_EVENTS:
                    result = self._observation_tools.get_recent_events()
                case AgentToolName.PROMOTE_ROLLOUT:
                    result = self._mutation_tools.promote_rollout()
                case AgentToolName.PAUSE_ROLLOUT:
                    result = self._mutation_tools.pause_rollout()
                case AgentToolName.ROLLBACK_ROLLOUT:
                    result = self._mutation_tools.rollback_rollout()
                case _:
                    raise AssertionError(f"unsupported agent tool: {call.tool!r}")
        except MutationLimitExceeded as exc:
            return AgentToolInteraction.failed(call, f"mutation limit exceeded: {exc}")

        return AgentToolInteraction.successful(call, _encode_tool_result(result))


def _encode_tool_result(result: BaseModel | tuple[BaseModel, ...]) -> str:
    """Copy a concrete tool value into immutable JSON text for model history."""
    if isinstance(result, tuple):
        payload = [item.model_dump(mode="json") for item in result]
    else:
        payload = result.model_dump(mode="json")
    return dumps(payload, separators=(",", ":"), sort_keys=True)


class AgentRunner:
    """Coordinate one run; each model output, tool call or final, uses one step."""

    def __init__(
        self,
        model: AgentModel,
        observation_tools: RolloutObservationTools,
        mutation_tools: RolloutMutationTools,
        max_steps: int = 8,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self._model = model
        self._dispatcher = RolloutToolDispatcher(observation_tools, mutation_tools)
        self._max_steps = max_steps

    def run(self, task: AgentTask) -> AgentRunResult:
        """Run until a final answer or until the model-output step limit."""
        interactions: tuple[AgentToolInteraction, ...] = ()

        for step in range(1, self._max_steps + 1):
            output: AgentOutput = self._model.respond(task, interactions)
            if isinstance(output, AgentFinalAnswer):
                return AgentRunResult(
                    final_answer=output,
                    interactions=interactions,
                    steps=step,
                )

            interaction = self._dispatcher.execute(output)
            interactions = (*interactions, interaction)

        raise AgentStepLimitExceeded(
            f"no final answer produced within {self._max_steps} model steps"
        )
