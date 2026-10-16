import json
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
from rollouteval.agent.providers import AgentProviderResponseError, GroqResponsesModel
from rollouteval.domain.scenario import Decision
from rollouteval.environment import RolloutEnvironment
from rollouteval.evaluation import GroundednessJudge, GroundednessJudgment, GraderStatus
from rollouteval.evaluation.evidence import EvidenceId, EvidenceStatus
from rollouteval.evaluation.grounding import GroundingContext, GroundingEvidenceItem
from rollouteval.evaluation.providers import (
    GroundednessJudgeResponseError,
    GroqGroundednessJudge,
)
from rollouteval.scenarios import load_scenario
from rollouteval.tools import RolloutMutationTools, RolloutObservationTools


SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"


class FakeResponses:
    def __init__(self, outputs=(), error=None):
        self.outputs = iter(outputs)
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return next(self.outputs)


class FakeClient:
    def __init__(self, outputs=(), error=None):
        self.responses = FakeResponses(outputs, error)


def _grounding_context() -> GroundingContext:
    return GroundingContext(
        task="Assess only observed rollout evidence.",
        decision=Decision.PAUSE,
        explanation="The rollout state is progressing.",
        evidence=(
            GroundingEvidenceItem(
                interaction_index=0,
                evidence_id=EvidenceId.ROLLOUT_STATE,
                source_tool="get_rollout_state",
                status=EvidenceStatus.OBSERVED,
                content='{"phase":"Progressing"}',
            ),
        ),
    )


def _grounding_judgment(status: str) -> str:
    return json.dumps({"status": status, "reason": "The supplied evidence supports this."})


@pytest.mark.parametrize("status", [status.value for status in GraderStatus])
def test_groq_groundedness_judge_validates_all_statuses(status):
    client = FakeClient([SimpleNamespace(output_text=_grounding_judgment(status))])
    judge = GroqGroundednessJudge(model="judge-model", client=client)

    assert isinstance(judge, GroundednessJudge)
    assert judge.provider_name == "groq"
    assert judge.model == "judge-model"
    assert len(judge.prompt_sha256) == 64
    result = judge.judge(_grounding_context())
    assert isinstance(result, GroundednessJudgment)
    assert result.status.value == status
    request = client.responses.calls[0]
    assert request["model"] == "judge-model"
    assert request["text"]["format"]["type"] == "json_schema"
    assert "store" not in request


@pytest.mark.parametrize("model", ["", "   ", "\n\t"])
def test_groq_groundedness_rejects_blank_model(model):
    with pytest.raises(ValueError, match="non-empty"):
        GroqGroundednessJudge(model=model, client=FakeClient())


def test_groq_groundedness_uses_injected_client_and_sends_bounded_context_only():
    context = _grounding_context()
    client = FakeClient([SimpleNamespace(output_text=_grounding_judgment("PASS"))])
    GroqGroundednessJudge(model="m", client=client).judge(context)
    payload = json.loads(client.responses.calls[0]["input"])

    assert payload == {
        "task": context.task,
        "decision": "pause",
        "explanation": context.explanation,
        "evidence": [{
            "interaction_index": 0,
            "evidence_id": "rollout_state",
            "source_tool": "get_rollout_state",
            "status": "OBSERVED",
            "content": '{"phase":"Progressing"}',
        }],
    }
    for forbidden in ("candidate_status", "expected_status", "reviewed_status", "reviewer", "reviewer_rationale"):
        assert forbidden not in json.dumps(payload)


@pytest.mark.parametrize(
    "output",
    ["{bad", json.dumps({"status": "MAYBE", "reason": "No."}), json.dumps({"status": "PASS", "reason": " "}), ""],
)
def test_groq_groundedness_rejects_malformed_or_invalid_output(output):
    client = FakeClient([SimpleNamespace(output_text=output)])
    judge = GroqGroundednessJudge(model="m", client=client)
    with pytest.raises(GroundednessJudgeResponseError):
        judge.judge(_grounding_context())
    assert len(client.responses.calls) == 1


def test_groq_groundedness_provider_error_propagates_without_retry():
    error = RuntimeError("provider failure")
    client = FakeClient(error=error)
    judge = GroqGroundednessJudge(model="m", client=client)
    with pytest.raises(RuntimeError) as caught:
        judge.judge(_grounding_context())
    assert caught.value is error
    assert len(client.responses.calls) == 1


def test_groq_groundedness_default_client_uses_groq_key_and_exact_base_url(monkeypatch):
    captured = {}
    fake = FakeClient([SimpleNamespace(output_text=_grounding_judgment("PASS"))])
    monkeypatch.setenv("GROQ_API_KEY", "dummy-groq-key")
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: captured.update(kwargs) or fake)

    GroqGroundednessJudge(model="m").judge(_grounding_context())

    assert captured == {
        "api_key": "dummy-groq-key",
        "base_url": "https://api.groq.com/openai/v1",
    }


def _agent_output(kind="final"):
    if kind == "tool_call":
        return json.dumps({"kind": kind, "tool": "get_service_metrics", "arguments": {}})
    return json.dumps({"kind": kind, "decision": "pause", "explanation": "Wait for evidence."})


@pytest.mark.parametrize("model", ["", "  ", "\t"])
def test_groq_agent_rejects_blank_model(model):
    with pytest.raises(ValueError, match="non-empty"):
        GroqResponsesModel(model=model, client=FakeClient())


def test_groq_agent_parses_existing_tool_call_and_final_contracts():
    client = FakeClient([
        SimpleNamespace(output_text=_agent_output("tool_call")),
        SimpleNamespace(output_text=_agent_output("final")),
    ])
    model = GroqResponsesModel(model="agent-model", client=client)

    tool_response = model.respond(AgentTask(task="Assess rollout."), ())
    final_response = model.respond(AgentTask(task="Assess rollout."), ())

    assert isinstance(tool_response, AgentModelResponse)
    assert isinstance(tool_response.output, AgentToolCall)
    assert tool_response.output.tool is AgentToolName.GET_SERVICE_METRICS
    assert client.responses.calls[0]["model"] == "agent-model"
    assert isinstance(final_response.output, AgentFinalAnswer)
    assert final_response.output.decision is Decision.PAUSE
    assert len(client.responses.calls) == 2


def test_groq_agent_maps_metadata_and_does_not_leak_provider_response():
    provider_response = SimpleNamespace(
        output_text=_agent_output(),
        model="actual-model-id",
        usage=SimpleNamespace(input_tokens=7, output_tokens=3, total_tokens=10),
    )
    client = FakeClient([provider_response])
    result = GroqResponsesModel(model="requested-model", client=client).respond(
        AgentTask(task="Assess rollout."), ()
    )

    assert result.model == "actual-model-id"
    assert result.usage == ModelUsage(
        input_tokens=7,
        output_tokens=3,
        total_tokens=10,
        cached_input_tokens=None,
        cache_write_tokens=None,
        reasoning_tokens=None,
    )
    assert result is not provider_response
    assert not hasattr(result, "output_text")


@pytest.mark.parametrize("output", ["{bad", '{"kind":"other"}', ""])
def test_groq_agent_invalid_output_fails_without_retry(output):
    client = FakeClient([SimpleNamespace(output_text=output)])
    model = GroqResponsesModel(model="m", client=client)
    with pytest.raises(AgentProviderResponseError):
        model.respond(AgentTask(task="Assess rollout."), ())
    assert len(client.responses.calls) == 1


def test_groq_agent_provider_exception_propagates_unchanged():
    error = RuntimeError("transport error")
    client = FakeClient(error=error)
    with pytest.raises(RuntimeError) as caught:
        GroqResponsesModel(model="m", client=client).respond(AgentTask(task="Task."), ())
    assert caught.value is error
    assert len(client.responses.calls) == 1


def test_groq_agent_runs_through_unchanged_agent_runner():
    scenario = load_scenario(SCENARIOS / "healthy-canary.yaml")
    environment = RolloutEnvironment(scenario)
    client = FakeClient([SimpleNamespace(output_text=_agent_output())])
    model = GroqResponsesModel(model="m", client=client)
    runner = AgentRunner(
        model,
        RolloutObservationTools(environment),
        RolloutMutationTools(environment),
    )

    result = runner.run(AgentTask.from_scenario(scenario))

    assert result.final_answer.decision is Decision.PAUSE
    assert result.steps == 1


def test_groq_agent_default_client_uses_groq_key_and_exact_base_url(monkeypatch):
    captured = {}
    fake = FakeClient([SimpleNamespace(output_text=_agent_output())])
    monkeypatch.setenv("GROQ_API_KEY", "dummy-groq-key")
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: captured.update(kwargs) or fake)

    GroqResponsesModel(model="m").respond(AgentTask(task="Task."), ())

    assert captured == {
        "api_key": "dummy-groq-key",
        "base_url": "https://api.groq.com/openai/v1",
    }
