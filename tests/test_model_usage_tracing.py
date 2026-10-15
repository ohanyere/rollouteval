from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from rollouteval.agent import (
    AgentFinalAnswer,
    AgentModelResponse,
    AgentRunner,
    AgentTask,
    AgentToolCall,
    AgentToolName,
    ModelUsage,
)
from rollouteval.agent.providers import OpenAIResponsesModel
from rollouteval.domain.scenario import Decision
from rollouteval.environment import RolloutEnvironment
from rollouteval.observability import TraceEventType, TraceRecorder
from rollouteval.scenarios import load_scenario
from rollouteval.tools import RolloutMutationTools, RolloutObservationTools

SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"


def _answer() -> AgentFinalAnswer:
    return AgentFinalAnswer(kind="final", decision=Decision.PROMOTE, explanation="Observed.")


def _call() -> AgentToolCall:
    return AgentToolCall(kind="tool_call", tool=AgentToolName.GET_ROLLOUT_STATE)


@pytest.mark.parametrize(
    "field",
    [
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
    ],
)
def test_model_usage_rejects_negative_counts(field: str) -> None:
    counts = {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
    counts[field] = -1
    with pytest.raises(ValidationError):
        ModelUsage(**counts)


def test_model_usage_accepts_counts_zero_and_unknown_details() -> None:
    usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)

    assert usage.input_tokens == 0
    assert usage.cached_input_tokens is None
    assert usage.cache_write_tokens is None
    assert usage.reasoning_tokens is None


def test_model_usage_is_immutable() -> None:
    usage = ModelUsage(input_tokens=1, output_tokens=2, total_tokens=3)
    with pytest.raises(ValidationError):
        usage.total_tokens = 4


@pytest.mark.parametrize("output", [_call(), _answer()])
def test_model_response_wraps_either_semantic_output(output: object) -> None:
    response = AgentModelResponse(output=output)
    assert response.output is output
    assert response.model is None
    assert response.usage is None


def test_model_response_rejects_blank_model_and_is_immutable() -> None:
    with pytest.raises(ValidationError):
        AgentModelResponse(output=_answer(), model="  ")
    response = AgentModelResponse(output=_answer(), model="model-x")
    with pytest.raises(ValidationError):
        response.model = "other"


class ScriptedModel:
    def __init__(self, responses: list[AgentModelResponse]) -> None:
        self.responses = iter(responses)
        self.histories = []

    def respond(self, task, interactions):
        self.histories.append(interactions)
        return next(self.responses)


class FakeClock:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return next(self.values)


def _runner(model, scenario_id: str = "healthy-canary", clock=None):
    scenario = load_scenario(SCENARIOS / f"{scenario_id}.yaml")
    environment = RolloutEnvironment(scenario)
    runner = AgentRunner(
        model,
        RolloutObservationTools(environment),
        RolloutMutationTools(environment),
        clock=clock if clock is not None else FakeClock([]),
    )
    return runner, environment, AgentTask.from_scenario(scenario)


def test_model_metadata_is_attached_to_matching_completed_trace_event() -> None:
    usage = ModelUsage(input_tokens=10, output_tokens=4, total_tokens=14, cached_input_tokens=0)
    model = ScriptedModel([AgentModelResponse(output=_answer(), model="actual-model", usage=usage)])
    clock = FakeClock([0, 1, 1.25, 2])
    runner, _, task = _runner(model, clock=clock)
    recorder = TraceRecorder("run-usage", "healthy-canary")

    result = runner.run(task, trace_recorder=recorder)
    events = recorder.snapshot().events
    completed = next(event for event in events if event.event_type is TraceEventType.MODEL_COMPLETED)

    assert result.steps == 1
    assert completed.step == 1
    assert completed.model == "actual-model"
    assert completed.model_usage == usage
    assert completed.duration_ms == 250
    assert events[-1].duration_ms == 2000
    assert [event.event_type for event in events] == [
        TraceEventType.RUN_STARTED,
        TraceEventType.MODEL_STARTED,
        TraceEventType.MODEL_COMPLETED,
        TraceEventType.RUN_COMPLETED,
    ]


def test_model_calls_have_independent_usage_and_no_semantic_usage_fields() -> None:
    first_usage = ModelUsage(input_tokens=3, output_tokens=1, total_tokens=4)
    second_usage = ModelUsage(input_tokens=7, output_tokens=2, total_tokens=9)
    model = ScriptedModel(
        [
            AgentModelResponse(output=_call(), model="m", usage=first_usage),
            AgentModelResponse(output=_answer(), model="m", usage=second_usage),
        ]
    )
    # run, model, model, tool, model, model, run
    clock = FakeClock([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    runner, _, task = _runner(model, clock=clock)
    recorder = TraceRecorder("run-multi-usage", "healthy-canary")

    result = runner.run(task, trace_recorder=recorder)
    events = recorder.snapshot().events
    completions = [event for event in events if event.event_type is TraceEventType.MODEL_COMPLETED]

    assert [event.model_usage for event in completions] == [first_usage, second_usage]
    assert [event.step for event in completions] == [1, 2]
    assert all(event.model_usage is None for event in events if event.event_type in {
        TraceEventType.TOOL_STARTED,
        TraceEventType.TOOL_COMPLETED,
    })
    assert len(result.interactions) == 1
    assert not hasattr(result, "usage")
    assert model.histories == [(), result.interactions]


def test_without_trace_model_usage_does_not_change_semantic_result_or_state() -> None:
    scenario = load_scenario(SCENARIOS / "healthy-canary.yaml")
    plain_environment = RolloutEnvironment(scenario)
    wrapped_environment = RolloutEnvironment(scenario)
    answer = _answer()
    usage = ModelUsage(input_tokens=5, output_tokens=1, total_tokens=6)
    plain_runner = AgentRunner(
        ScriptedModel([AgentModelResponse(output=answer)]),
        RolloutObservationTools(plain_environment),
        RolloutMutationTools(plain_environment),
    )
    usage_runner = AgentRunner(
        ScriptedModel([AgentModelResponse(output=answer, usage=usage)]),
        RolloutObservationTools(wrapped_environment),
        RolloutMutationTools(wrapped_environment),
    )

    plain_result = plain_runner.run(AgentTask(task=scenario.task))
    usage_result = usage_runner.run(AgentTask(task=scenario.task))

    assert plain_result == usage_result
    assert plain_environment.snapshot() == wrapped_environment.snapshot()
    assert plain_environment.mutation_count == wrapped_environment.mutation_count
    assert "usage" not in AgentFinalAnswer.model_fields


def test_trace_metadata_is_restricted_to_model_completed() -> None:
    usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
    recorder = TraceRecorder("run-restrictions", "scenario")
    event = recorder.record(
        TraceEventType.MODEL_COMPLETED, step=1, model="m", model_usage=usage
    )
    assert event.model == "m"
    assert event.model_usage == usage
    assert recorder.record(TraceEventType.MODEL_COMPLETED, step=2).model_usage is None

    for event_type, step, tool_name in (
        (TraceEventType.MODEL_STARTED, 1, None),
        (TraceEventType.MODEL_FAILED, 1, None),
        (TraceEventType.TOOL_COMPLETED, 1, AgentToolName.GET_ROLLOUT_STATE),
        (TraceEventType.RUN_COMPLETED, None, None),
    ):
        with pytest.raises(ValidationError, match="model metadata"):
            recorder.record(event_type, step=step, tool_name=tool_name, model_usage=usage)

    with pytest.raises(ValidationError, match="model metadata"):
        recorder.record(TraceEventType.RUN_STARTED, model="m")


def test_openai_provider_maps_usage_and_prefers_returned_model_identifier() -> None:
    response = SimpleNamespace(
        output_text='{"kind":"final","decision":"promote","explanation":"Observed."}',
        model="resolved-model-2026-10-01",
        usage=SimpleNamespace(
            input_tokens=21,
            output_tokens=8,
            total_tokens=29,
            input_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=4),
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
        ),
    )

    class FakeResponses:
        def create(self, **kwargs):
            return response

    provider = OpenAIResponsesModel("configured-model", client=SimpleNamespace(responses=FakeResponses()))
    result = provider.respond(AgentTask(task="Assess rollout."), ())

    assert result.model == "resolved-model-2026-10-01"
    assert isinstance(result.output, AgentFinalAnswer)
    assert result.usage == ModelUsage(
        input_tokens=21,
        output_tokens=8,
        total_tokens=29,
        cached_input_tokens=0,
        cache_write_tokens=4,
        reasoning_tokens=3,
    )


def test_openai_provider_uses_configured_model_and_none_usage_when_absent() -> None:
    response = SimpleNamespace(
        output_text='{"kind":"final","decision":"promote","explanation":"Observed."}',
        usage=None,
    )

    class FakeResponses:
        def create(self, **kwargs):
            return response

    provider = OpenAIResponsesModel("configured-model", client=SimpleNamespace(responses=FakeResponses()))
    result = provider.respond(AgentTask(task="Assess rollout."), ())

    assert result.model == "configured-model"
    assert result.usage is None


def test_model_usage_has_no_evaluator_truth_or_provider_objects() -> None:
    usage = ModelUsage(input_tokens=1, output_tokens=2, total_tokens=3)
    response = AgentModelResponse(output=_answer(), model="m", usage=usage)

    assert set(ModelUsage.model_fields) == {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
    }
    assert "scenario" not in AgentModelResponse.model_fields
    assert "client" not in AgentModelResponse.model_fields
    assert not {"expected_decision", "expected_final_state"} & set(ModelUsage.model_fields)
    assert response.usage is usage
