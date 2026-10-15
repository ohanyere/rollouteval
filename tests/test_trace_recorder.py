from pydantic import ValidationError
import pytest

from rollouteval.agent import (
    AgentFinalAnswer,
    AgentRunResult,
    AgentToolName,
)
from rollouteval.domain.scenario import Decision
from rollouteval.observability import RunTrace, TraceEvent, TraceEventType, TraceRecorder


EVENT_TYPES = {
    "run.started",
    "model.started",
    "model.completed",
    "model.failed",
    "tool.started",
    "tool.completed",
    "tool.failed",
    "run.completed",
    "run.failed",
}


def _event(
    sequence: int = 0,
    event_type: TraceEventType = TraceEventType.RUN_STARTED,
    *,
    step: int | None = None,
    tool_name: AgentToolName | None = None,
    message: str | None = None,
) -> TraceEvent:
    return TraceEvent(
        sequence=sequence,
        event_type=event_type,
        step=step,
        tool_name=tool_name,
        message=message,
    )


def test_trace_event_type_contains_exactly_nine_v1_values() -> None:
    assert {event_type.value for event_type in TraceEventType} == EVENT_TYPES


def test_trace_event_represents_run_started() -> None:
    assert _event().event_type is TraceEventType.RUN_STARTED


def test_trace_event_represents_model_started_with_step() -> None:
    event = _event(event_type=TraceEventType.MODEL_STARTED, step=1)

    assert event.step == 1
    assert event.tool_name is None


def test_trace_event_represents_tool_started_with_step_and_tool() -> None:
    event = _event(
        event_type=TraceEventType.TOOL_STARTED,
        step=1,
        tool_name=AgentToolName.GET_ROLLOUT_STATE,
    )

    assert event.tool_name is AgentToolName.GET_ROLLOUT_STATE


@pytest.mark.parametrize("sequence", [-1, -10])
def test_trace_event_rejects_negative_sequence(sequence: int) -> None:
    with pytest.raises(ValidationError):
        _event(sequence=sequence)


def test_trace_event_rejects_zero_step() -> None:
    with pytest.raises(ValidationError):
        _event(event_type=TraceEventType.MODEL_STARTED, step=0)


def test_tool_event_requires_tool_name() -> None:
    with pytest.raises(ValidationError, match="tool events require tool_name"):
        _event(event_type=TraceEventType.TOOL_STARTED)


def test_non_tool_event_rejects_tool_name() -> None:
    with pytest.raises(ValidationError, match="non-tool events"):
        _event(event_type=TraceEventType.MODEL_STARTED, tool_name=AgentToolName.PROMOTE_ROLLOUT)


def test_trace_event_rejects_whitespace_only_message() -> None:
    with pytest.raises(ValidationError):
        _event(message="   \t")


def test_trace_event_is_immutable() -> None:
    event = _event()

    with pytest.raises(ValidationError):
        event.sequence = 8


def test_run_trace_accepts_non_empty_identity() -> None:
    trace = RunTrace(run_id="run-001", scenario_id="degraded-canary", events=())

    assert trace.run_id == "run-001"
    assert trace.scenario_id == "degraded-canary"


@pytest.mark.parametrize("run_id", ["", "  "])
def test_run_trace_rejects_empty_run_id(run_id: str) -> None:
    with pytest.raises(ValidationError):
        RunTrace(run_id=run_id, scenario_id="degraded-canary", events=())


@pytest.mark.parametrize("scenario_id", ["", "  "])
def test_run_trace_rejects_empty_scenario_id(scenario_id: str) -> None:
    with pytest.raises(ValidationError):
        RunTrace(run_id="run-001", scenario_id=scenario_id, events=())


def test_run_trace_events_are_immutable_and_detached_from_input_list() -> None:
    source_events = [_event()]
    trace = RunTrace(run_id="run-001", scenario_id="scenario", events=source_events)
    source_events.append(_event(sequence=1, event_type=TraceEventType.RUN_FAILED))

    assert len(trace.events) == 1
    with pytest.raises(ValidationError):
        trace.events[0].sequence = 9


def test_run_trace_contract_has_no_evaluator_private_fields() -> None:
    assert set(RunTrace.model_fields) == {"run_id", "scenario_id", "events"}
    assert not {
        "expected_decision",
        "expected_final_state",
        "required_evidence",
        "max_mutations",
    } & set(RunTrace.model_fields)


def test_recorder_starts_with_empty_history_without_auto_event() -> None:
    recorder = TraceRecorder(run_id="run-001", scenario_id="degraded-canary")

    assert recorder.snapshot().events == ()


def test_first_and_second_events_receive_zero_and_one_sequences() -> None:
    recorder = TraceRecorder(run_id="run-001", scenario_id="degraded-canary")

    first = recorder.record(TraceEventType.RUN_STARTED)
    second = recorder.record(TraceEventType.MODEL_STARTED, step=1)

    assert (first.sequence, second.sequence) == (0, 1)


def test_recorder_preserves_insertion_order_and_returns_stored_event() -> None:
    recorder = TraceRecorder(run_id="run-001", scenario_id="degraded-canary")
    first = recorder.record(TraceEventType.RUN_STARTED)
    second = recorder.record(TraceEventType.MODEL_STARTED, step=1)

    trace = recorder.snapshot()

    assert trace.events == (first, second)
    assert trace.events[0] is first


def test_snapshot_contains_current_events_and_recorder_identity() -> None:
    recorder = TraceRecorder(run_id="run-001", scenario_id="healthy-canary")
    recorder.record(TraceEventType.RUN_STARTED)

    trace = recorder.snapshot()

    assert trace.run_id == "run-001"
    assert trace.scenario_id == "healthy-canary"
    assert len(trace.events) == 1


def test_snapshot_does_not_expose_mutable_internal_event_storage() -> None:
    recorder = TraceRecorder(run_id="run-001", scenario_id="degraded-canary")
    recorder.record(TraceEventType.RUN_STARTED)
    snapshot = recorder.snapshot()

    with pytest.raises(AttributeError):
        snapshot.events.append(_event(sequence=1))  # type: ignore[attr-defined]


def test_earlier_snapshot_does_not_change_after_later_event() -> None:
    recorder = TraceRecorder(run_id="run-001", scenario_id="degraded-canary")
    recorder.record(TraceEventType.RUN_STARTED)
    earlier = recorder.snapshot()
    recorder.record(TraceEventType.RUN_COMPLETED)
    later = recorder.snapshot()

    assert len(earlier.events) == 1
    assert len(later.events) == 2
    assert earlier.events[0].event_type is TraceEventType.RUN_STARTED


def test_separate_recorders_do_not_share_events() -> None:
    first = TraceRecorder(run_id="run-001", scenario_id="scenario-a")
    second = TraceRecorder(run_id="run-002", scenario_id="scenario-b")
    first.record(TraceEventType.RUN_STARTED)

    assert len(first.snapshot().events) == 1
    assert second.snapshot().events == ()


def test_caller_cannot_choose_sequence_number() -> None:
    recorder = TraceRecorder(run_id="run-001", scenario_id="scenario")

    with pytest.raises(TypeError):
        recorder.record(TraceEventType.RUN_STARTED, sequence=12)  # type: ignore[call-arg]


def test_model_events_can_be_recorded_without_tool_name() -> None:
    recorder = TraceRecorder(run_id="run-001", scenario_id="scenario")

    event = recorder.record(TraceEventType.MODEL_COMPLETED, step=1)

    assert event.tool_name is None


@pytest.mark.parametrize("tool", list(AgentToolName))
def test_all_agent_tools_can_appear_in_tool_events(tool: AgentToolName) -> None:
    recorder = TraceRecorder(run_id="run-001", scenario_id="scenario")

    event = recorder.record(
        TraceEventType.TOOL_STARTED, step=1, tool_name=tool
    )

    assert event.tool_name is tool


def test_recorder_represents_successful_run_lifecycle_in_order() -> None:
    recorder = TraceRecorder(run_id="run-001", scenario_id="degraded-canary")
    recorder.record(TraceEventType.RUN_STARTED)
    recorder.record(TraceEventType.MODEL_STARTED, step=1)
    recorder.record(TraceEventType.MODEL_COMPLETED, step=1)
    recorder.record(
        TraceEventType.TOOL_STARTED,
        step=1,
        tool_name=AgentToolName.GET_ROLLOUT_STATE,
    )
    recorder.record(
        TraceEventType.TOOL_COMPLETED,
        step=1,
        tool_name=AgentToolName.GET_ROLLOUT_STATE,
    )
    recorder.record(TraceEventType.MODEL_STARTED, step=2)
    recorder.record(TraceEventType.MODEL_COMPLETED, step=2)
    recorder.record(TraceEventType.RUN_COMPLETED)

    trace = recorder.snapshot()

    assert [event.event_type for event in trace.events] == [
        TraceEventType.RUN_STARTED,
        TraceEventType.MODEL_STARTED,
        TraceEventType.MODEL_COMPLETED,
        TraceEventType.TOOL_STARTED,
        TraceEventType.TOOL_COMPLETED,
        TraceEventType.MODEL_STARTED,
        TraceEventType.MODEL_COMPLETED,
        TraceEventType.RUN_COMPLETED,
    ]
    assert [event.sequence for event in trace.events] == list(range(8))


def test_recorder_represents_model_failure_lifecycle():
    recorder = TraceRecorder(run_id="run-001", scenario_id="scenario")
    recorder.record(TraceEventType.RUN_STARTED)
    recorder.record(TraceEventType.MODEL_STARTED, step=1)
    recorder.record(TraceEventType.MODEL_FAILED, step=1, message="provider failed")
    recorder.record(TraceEventType.RUN_FAILED, message="run failed")

    assert [event.event_type for event in recorder.snapshot().events] == [
        TraceEventType.RUN_STARTED,
        TraceEventType.MODEL_STARTED,
        TraceEventType.MODEL_FAILED,
        TraceEventType.RUN_FAILED,
    ]


def test_recorder_represents_tool_failure_lifecycle():
    recorder = TraceRecorder(run_id="run-001", scenario_id="scenario")
    recorder.record(
        TraceEventType.TOOL_STARTED,
        step=1,
        tool_name=AgentToolName.GET_ROLLOUT_STATE,
    )
    recorder.record(
        TraceEventType.TOOL_FAILED,
        step=1,
        tool_name=AgentToolName.GET_ROLLOUT_STATE,
        message="tool failed",
    )

    assert [event.event_type for event in recorder.snapshot().events] == [
        TraceEventType.TOOL_STARTED,
        TraceEventType.TOOL_FAILED,
    ]


def test_recording_trace_does_not_mutate_agent_run_result():
    run = AgentRunResult(
        final_answer=AgentFinalAnswer(
            kind="final", decision=Decision.PAUSE, explanation="test"
        ),
        interactions=(),
        steps=1,
    )
    before = run.model_dump(mode="json")
    recorder = TraceRecorder(run_id="run-001", scenario_id="scenario")
    recorder.record(TraceEventType.RUN_STARTED)
    recorder.record(TraceEventType.RUN_COMPLETED)

    assert run.model_dump(mode="json") == before


def test_trace_contracts_have_no_scenario_reference_or_evaluation_truth():
    assert "Scenario" not in repr(TraceEvent.model_fields)
    assert "Scenario" not in repr(RunTrace.model_fields)
    for field in (
        "expected_decision",
        "expected_final_state",
        "required_evidence",
        "max_mutations",
    ):
        assert field not in TraceEvent.model_fields
        assert field not in RunTrace.model_fields
