"""Immutable operational trace contracts and an in-memory recorder."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rollouteval.agent.contracts import AgentToolName


class TraceEventType(StrEnum):
    RUN_STARTED = "run.started"
    MODEL_STARTED = "model.started"
    MODEL_COMPLETED = "model.completed"
    MODEL_FAILED = "model.failed"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"


_TOOL_EVENT_TYPES = frozenset(
    {
        TraceEventType.TOOL_STARTED,
        TraceEventType.TOOL_COMPLETED,
        TraceEventType.TOOL_FAILED,
    }
)


class TraceEvent(BaseModel):
    """One immutable operational event in a run trace."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    sequence: int = Field(ge=0)
    event_type: TraceEventType
    step: int | None = Field(default=None, ge=1)
    tool_name: AgentToolName | None = None
    message: str | None = Field(default=None, min_length=1)
    duration_ms: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_tool_event_fields(self) -> "TraceEvent":
        is_tool_event = self.event_type in _TOOL_EVENT_TYPES
        if is_tool_event and self.tool_name is None:
            raise ValueError("tool events require tool_name")
        if not is_tool_event and self.tool_name is not None:
            raise ValueError("non-tool events must not include tool_name")
        duration_event_types = {
            TraceEventType.MODEL_COMPLETED,
            TraceEventType.MODEL_FAILED,
            TraceEventType.TOOL_COMPLETED,
            TraceEventType.TOOL_FAILED,
            TraceEventType.RUN_COMPLETED,
            TraceEventType.RUN_FAILED,
        }
        if self.duration_ms is not None and self.event_type not in duration_event_types:
            raise ValueError("duration_ms is only valid on completed or failed events")
        return self


class RunTrace(BaseModel):
    """Immutable trace identity and ordered event snapshot for one run."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    run_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    events: tuple[TraceEvent, ...]


class TraceRecorder:
    """Assign sequence numbers and retain immutable events in memory."""

    def __init__(self, run_id: str, scenario_id: str) -> None:
        identity = RunTrace(run_id=run_id, scenario_id=scenario_id, events=())
        self._run_id = identity.run_id
        self._scenario_id = identity.scenario_id
        self._events: list[TraceEvent] = []

    def record(
        self,
        event_type: TraceEventType,
        *,
        step: int | None = None,
        tool_name: AgentToolName | None = None,
        message: str | None = None,
        duration_ms: float | None = None,
    ) -> TraceEvent:
        """Validate and store one event with the next recorder-owned sequence."""
        event = TraceEvent(
            sequence=len(self._events),
            event_type=event_type,
            step=step,
            tool_name=tool_name,
            message=message,
            duration_ms=duration_ms,
        )
        self._events.append(event)
        return event

    def snapshot(self) -> RunTrace:
        """Return a detached immutable view of events recorded so far."""
        return RunTrace(
            run_id=self._run_id,
            scenario_id=self._scenario_id,
            events=tuple(self._events),
        )
