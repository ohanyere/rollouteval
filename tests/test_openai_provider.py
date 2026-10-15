import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rollouteval.agent import (
    AgentFinalAnswer,
    AgentModelResponse,
    AgentRunner,
    AgentTask,
    AgentToolCall,
    AgentToolInteraction,
    AgentToolName,
)
from rollouteval.agent.providers import AgentProviderResponseError, OpenAIResponsesModel
from rollouteval.domain.scenario import Decision
from rollouteval.environment import RolloutEnvironment
from rollouteval.scenarios import load_scenario
from rollouteval.tools import RolloutMutationTools, RolloutObservationTools


SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"
SCENARIO_TOOL_NAMES = tuple(AgentToolName)


class FakeResponses:
    def __init__(self, outputs: list[str | Exception | SimpleNamespace]) -> None:
        self.outputs = iter(outputs)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        if isinstance(output, SimpleNamespace):
            return output
        return SimpleNamespace(output_text=output)


class FakeClient:
    def __init__(self, outputs: list[str | Exception]) -> None:
        self.responses = FakeResponses(outputs)


def _tool_json(tool: AgentToolName) -> str:
    return json.dumps({"kind": "tool_call", "tool": tool.value, "arguments": {}})


def _final_json(decision: str = "pause", explanation: str = "Wait for more evidence.") -> str:
    return json.dumps({"kind": "final", "decision": decision, "explanation": explanation})


def _provider(
    outputs: list[str | Exception], model: str = "test-model"
) -> tuple[OpenAIResponsesModel, FakeClient]:
    client = FakeClient(outputs)
    return OpenAIResponsesModel(model=model, client=client), client


def _request_data(request: dict[str, object]) -> dict[str, object]:
    return json.loads(request["input"])


def test_provider_constructs_with_non_empty_model() -> None:
    provider, _ = _provider([_final_json()], model="model-identifier")

    assert provider.model == "model-identifier"


@pytest.mark.parametrize("model", ["", "   ", "\n\t"])
def test_provider_rejects_empty_model_identifier(model: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        OpenAIResponsesModel(model=model, client=FakeClient([]))


def test_injected_client_does_not_require_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider, _ = _provider([_final_json()])

    assert isinstance(provider, OpenAIResponsesModel)


def test_provider_request_contains_task_text() -> None:
    provider, client = _provider([_final_json()])

    provider.respond(AgentTask(task="Assess candidate latency."), ())

    assert _request_data(client.responses.calls[0])["task"] == "Assess candidate latency."


def test_provider_instructions_name_all_six_allowed_tools() -> None:
    provider, client = _provider([_final_json()])

    provider.respond(AgentTask(task="Assess rollout."), ())

    instructions = client.responses.calls[0]["instructions"]
    assert all(tool.value in instructions for tool in SCENARIO_TOOL_NAMES)


def test_provider_serializes_history_in_chronological_order() -> None:
    provider, client = _provider([_final_json()])
    first = AgentToolInteraction.successful(
        AgentToolCall(kind="tool_call", tool=AgentToolName.GET_ROLLOUT_STATE),
        '{"phase":"Progressing"}',
    )
    second = AgentToolInteraction.failed(
        AgentToolCall(kind="tool_call", tool=AgentToolName.PROMOTE_ROLLOUT),
        "mutation limit exceeded",
    )

    provider.respond(AgentTask(task="Assess rollout."), (first, second))

    history = _request_data(client.responses.calls[0])["interaction_history"]
    assert [item["tool"] for item in history] == ["get_rollout_state", "promote_rollout"]
    assert history[0]["result"] == '{"phase":"Progressing"}'
    assert history[0]["error"] is None
    assert history[1]["result"] is None
    assert history[1]["error"] == "mutation limit exceeded"


def test_provider_serializes_successful_result_without_reinterpretation() -> None:
    provider, client = _provider([_final_json()])
    interaction = AgentToolInteraction.successful(
        AgentToolCall(kind="tool_call", tool=AgentToolName.GET_SERVICE_METRICS),
        '{"candidate_error_rate":0.087}',
    )

    provider.respond(AgentTask(task="Assess rollout."), (interaction,))

    history_item = _request_data(client.responses.calls[0])["interaction_history"][0]
    assert history_item["result"] == '{"candidate_error_rate":0.087}'


def test_provider_request_excludes_evaluator_private_fields() -> None:
    provider, client = _provider([_final_json()])
    interaction = AgentToolInteraction.successful(
        AgentToolCall(kind="tool_call", tool=AgentToolName.GET_ROLLOUT_STATE), "{}"
    )

    provider.respond(AgentTask(task="Assess rollout."), (interaction,))

    request_text = json.dumps(client.responses.calls[0])
    for private_field in (
        "expected_decision",
        "expected_final_state",
        "required_evidence",
        "max_mutations",
    ):
        assert private_field not in request_text


def test_provider_instructions_treat_tool_results_as_untrusted_data() -> None:
    provider, client = _provider([_final_json()])

    provider.respond(AgentTask(task="Assess rollout."), ())

    instructions = client.responses.calls[0]["instructions"].lower()
    assert "untrusted environment data" in instructions
    assert "not instructions" in instructions


def test_provider_uses_responses_json_mode_without_server_state() -> None:
    provider, client = _provider([_final_json()])

    provider.respond(AgentTask(task="Assess rollout."), ())

    request = client.responses.calls[0]
    assert request["model"] == "test-model"
    assert request["text"] == {"format": {"type": "json_object"}}
    assert "previous_response_id" not in request
    assert "conversation" not in request


@pytest.mark.parametrize("tool", SCENARIO_TOOL_NAMES)
def test_provider_parses_each_allowed_tool_call(tool: AgentToolName) -> None:
    provider, _ = _provider([_tool_json(tool)])

    output = provider.respond(AgentTask(task="Assess rollout."), ())

    assert isinstance(output, AgentModelResponse)
    assert isinstance(output.output, AgentToolCall)
    assert output.output.tool is tool


def test_provider_parses_valid_final_answer() -> None:
    provider, _ = _provider([_final_json()])

    output = provider.respond(AgentTask(task="Assess rollout."), ())

    assert isinstance(output, AgentModelResponse)
    assert isinstance(output.output, AgentFinalAnswer)
    assert output.output.decision is Decision.PAUSE


@pytest.mark.parametrize("decision", list(Decision))
def test_provider_parses_each_final_decision(decision: Decision) -> None:
    provider, _ = _provider([_final_json(decision.value)])

    output = provider.respond(AgentTask(task="Assess rollout."), ())

    assert isinstance(output, AgentModelResponse)
    assert isinstance(output.output, AgentFinalAnswer)
    assert output.output.decision is decision


def test_empty_response_raises_provider_response_error() -> None:
    provider, _ = _provider([""])

    with pytest.raises(AgentProviderResponseError, match="empty"):
        provider.respond(AgentTask(task="Assess rollout."), ())


def test_non_json_response_raises_provider_response_error_with_cause() -> None:
    provider, _ = _provider(["The answer is pause."])

    with pytest.raises(AgentProviderResponseError, match="valid JSON") as caught:
        provider.respond(AgentTask(task="Assess rollout."), ())

    assert isinstance(caught.value.__cause__, json.JSONDecodeError)


def test_unknown_tool_raises_provider_response_error_with_cause() -> None:
    provider, _ = _provider(['{"kind":"tool_call","tool":"delete_cluster","arguments":{}}'])

    with pytest.raises(AgentProviderResponseError, match="AgentOutput contract") as caught:
        provider.respond(AgentTask(task="Assess rollout."), ())

    assert caught.value.__cause__ is not None


def test_invalid_decision_raises_provider_response_error() -> None:
    provider, _ = _provider([_final_json("wait")])

    with pytest.raises(AgentProviderResponseError, match="AgentOutput contract"):
        provider.respond(AgentTask(task="Assess rollout."), ())


def test_undeclared_output_fields_raise_provider_response_error() -> None:
    provider, _ = _provider(
        ['{"kind":"final","decision":"pause","explanation":"Wait.","extra":true}']
    )

    with pytest.raises(AgentProviderResponseError):
        provider.respond(AgentTask(task="Assess rollout."), ())


def test_whitespace_explanation_raises_provider_response_error() -> None:
    provider, _ = _provider([_final_json("pause", "   ")])

    with pytest.raises(AgentProviderResponseError):
        provider.respond(AgentTask(task="Assess rollout."), ())


def test_malformed_provider_output_does_not_fabricate_fallback() -> None:
    provider, client = _provider(["not json"])

    with pytest.raises(AgentProviderResponseError):
        provider.respond(AgentTask(task="Assess rollout."), ())

    assert len(client.responses.calls) == 1


def test_openai_client_failure_propagates_original_exception() -> None:
    failure = RuntimeError("simulated provider failure")
    provider, _ = _provider([failure])

    with pytest.raises(RuntimeError) as caught:
        provider.respond(AgentTask(task="Assess rollout."), ())

    assert caught.value is failure
    assert caught.value.__cause__ is None


def test_first_provider_call_uses_empty_history() -> None:
    provider, client = _provider([_final_json()])

    provider.respond(AgentTask(task="Assess rollout."), ())

    assert _request_data(client.responses.calls[0])["interaction_history"] == []


def test_later_provider_call_includes_previous_interaction() -> None:
    provider, client = _provider([_tool_json(AgentToolName.GET_ROLLOUT_STATE), _final_json()])
    task = AgentTask(task="Assess rollout.")
    output = provider.respond(task, ())
    assert isinstance(output.output, AgentToolCall)
    history = (
        AgentToolInteraction.successful(output.output, '{"phase":"Progressing"}'),
    )

    provider.respond(task, history)

    assert _request_data(client.responses.calls[1])["interaction_history"][0]["result"] == (
        '{"phase":"Progressing"}'
    )


def test_mutation_limit_error_is_represented_in_provider_input() -> None:
    provider, client = _provider([_final_json()])
    interaction = AgentToolInteraction.failed(
        AgentToolCall(kind="tool_call", tool=AgentToolName.ROLLBACK_ROLLOUT),
        "mutation limit exceeded (1)",
    )

    provider.respond(AgentTask(task="Assess rollout."), (interaction,))

    history_item = _request_data(client.responses.calls[0])["interaction_history"][0]
    assert history_item["tool"] == "rollback_rollout"
    assert history_item["error"] == "mutation limit exceeded (1)"
    assert history_item["result"] is None


def test_openai_responses_model_satisfies_agent_model_protocol_shape() -> None:
    provider, _ = _provider([_final_json()])

    assert callable(provider.respond)
    response = provider.respond(AgentTask(task="Assess rollout."), ())
    assert isinstance(response, AgentModelResponse)
    assert isinstance(response.output, AgentFinalAnswer)


def test_provider_works_through_runner_with_fake_client_offline() -> None:
    scenario = load_scenario(SCENARIOS / "healthy-canary.yaml")
    environment = RolloutEnvironment(scenario)
    provider, client = _provider(
        [_tool_json(AgentToolName.GET_ROLLOUT_STATE), _final_json("promote", "Observed rollout state.")]
    )
    runner = AgentRunner(
        provider,
        RolloutObservationTools(environment),
        RolloutMutationTools(environment),
    )

    result = runner.run(AgentTask(task=scenario.task))

    assert result.final_answer.decision is Decision.PROMOTE
    assert len(result.interactions) == 1
    assert json.loads(result.interactions[0].result)["phase"] == "Progressing"
    assert len(client.responses.calls) == 2
    assert _request_data(client.responses.calls[1])["interaction_history"][0]["tool"] == (
        "get_rollout_state"
    )
