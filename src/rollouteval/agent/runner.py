"""Synchronous, bounded execution of model-requested rollout tools."""

from json import dumps
from time import perf_counter
from typing import Callable, Protocol

from pydantic import BaseModel

from rollouteval.agent.contracts import (
    AgentFinalAnswer,
    AgentModelResponse,
    AgentRunResult,
    AgentTask,
    AgentToolCall,
    AgentToolInteraction,
    AgentToolName,
)
from rollouteval.environment import MutationLimitExceeded
from rollouteval.observability import TraceEventType, TraceRecorder
from rollouteval.tools import RolloutMutationTools, RolloutObservationTools


class AgentModel(Protocol):
    """Provider-neutral interface for one synchronous model response."""

    def respond(
        self,
        task: AgentTask,
        interactions: tuple[AgentToolInteraction, ...],
    ) -> AgentModelResponse:
        """Return the next tool call or final answer."""
        ...


class AgentRunExecutor(Protocol):
    """Provider-neutral execution interface consumed by orchestration."""

    def run(
        self,
        task: AgentTask,
        *,
        trace_recorder: TraceRecorder | None = None,
    ) -> AgentRunResult:
        """Return a completed agent result or raise an execution error."""
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
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self._model = model
        self._dispatcher = RolloutToolDispatcher(observation_tools, mutation_tools)
        self._max_steps = max_steps
        self._clock = clock

    def run(
        self,
        task: AgentTask,
        *,
        trace_recorder: TraceRecorder | None = None,
    ) -> AgentRunResult:
        """Run until a final answer or until the model-output step limit."""
        interactions: tuple[AgentToolInteraction, ...] = ()
        run_start = self._clock() if trace_recorder is not None else None
        if trace_recorder is not None:
            trace_recorder.record(TraceEventType.RUN_STARTED)

        for step in range(1, self._max_steps + 1):
            if trace_recorder is not None:
                trace_recorder.record(TraceEventType.MODEL_STARTED, step=step)
                model_start = self._clock()
            try:
                model_response = self._model.respond(task, interactions)
            except Exception as exc:
                if trace_recorder is not None:
                    model_end = self._clock()
                    trace_recorder.record(
                        TraceEventType.MODEL_FAILED,
                        step=step,
                        message=f"Model failed: {type(exc).__name__}",
                        duration_ms=(model_end - model_start) * 1000,
                    )
                    run_end = self._clock()
                    trace_recorder.record(
                        TraceEventType.RUN_FAILED,
                        message=f"Run failed: {type(exc).__name__}",
                        duration_ms=(run_end - run_start) * 1000,
                    )
                raise

            if trace_recorder is not None:
                model_end = self._clock()
                trace_recorder.record(
                    TraceEventType.MODEL_COMPLETED,
                    step=step,
                    duration_ms=(model_end - model_start) * 1000,
                    model=model_response.model,
                    model_usage=model_response.usage,
                )
            output = model_response.output
            if isinstance(output, AgentFinalAnswer):
                if trace_recorder is not None:
                    run_end = self._clock()
                    trace_recorder.record(
                        TraceEventType.RUN_COMPLETED,
                        duration_ms=(run_end - run_start) * 1000,
                    )
                return AgentRunResult(
                    final_answer=output,
                    interactions=interactions,
                    steps=step,
                )

            if trace_recorder is not None:
                trace_recorder.record(
                    TraceEventType.TOOL_STARTED,
                    step=step,
                    tool_name=output.tool,
                )
                tool_start = self._clock()
            try:
                interaction = self._dispatcher.execute(output)
            except Exception as exc:
                if trace_recorder is not None:
                    tool_end = self._clock()
                    trace_recorder.record(
                        TraceEventType.TOOL_FAILED,
                        step=step,
                        tool_name=output.tool,
                        message=f"Tool failed: {type(exc).__name__}",
                        duration_ms=(tool_end - tool_start) * 1000,
                    )
                    run_end = self._clock()
                    trace_recorder.record(
                        TraceEventType.RUN_FAILED,
                        message=f"Run failed: {type(exc).__name__}",
                        duration_ms=(run_end - run_start) * 1000,
                    )
                raise

            if trace_recorder is not None:
                tool_end = self._clock()
                if interaction.error is None:
                    trace_recorder.record(
                        TraceEventType.TOOL_COMPLETED,
                        step=step,
                        tool_name=output.tool,
                        duration_ms=(tool_end - tool_start) * 1000,
                    )
                else:
                    trace_recorder.record(
                        TraceEventType.TOOL_FAILED,
                        step=step,
                        tool_name=output.tool,
                        message="mutation limit exceeded",
                        duration_ms=(tool_end - tool_start) * 1000,
                    )
            interactions = (*interactions, interaction)

        if trace_recorder is not None:
            run_end = self._clock()
            trace_recorder.record(
                TraceEventType.RUN_FAILED,
                message="Agent step limit exceeded",
                duration_ms=(run_end - run_start) * 1000,
            )
        raise AgentStepLimitExceeded(
            f"no final answer produced within {self._max_steps} model steps"
        )
