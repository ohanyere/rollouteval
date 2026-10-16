import importlib.util
import json
from pathlib import Path

import pytest

from rollouteval.agent import AgentTask, AgentToolName
from rollouteval.environment import RolloutEnvironment
from rollouteval.tools import RolloutMutationTools, RolloutObservationTools


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_live_agent.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("run_live_agent", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
run_live_agent = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(run_live_agent)


class FakeResponses:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = iter(outputs)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object):
        self.calls.append(kwargs)
        return type("FakeResponse", (), {"output_text": next(self._outputs)})()


class FakeClient:
    def __init__(self, outputs: list[str]) -> None:
        self.responses = FakeResponses(outputs)


def _response(kind: str, **fields: object) -> str:
    return json.dumps({"kind": kind, **fields})


def _fake_client() -> FakeClient:
    return FakeClient(
        [
            _response(
                "tool_call",
                tool=AgentToolName.ROLLBACK_ROLLOUT.value,
                arguments={},
            ),
            _response(
                "final",
                decision="rollback",
                explanation="The observed rollout should be rolled back.",
            ),
        ]
    )


def test_live_helper_loads_only_degraded_canary() -> None:
    scenario, _, _, _, _ = run_live_agent.build_live_run(
        "degraded-canary", "test-model", client=_fake_client(), provider="openai"
    )

    assert scenario.id == "degraded-canary"
    assert scenario.name


def test_live_helper_constructs_safe_agent_task() -> None:
    _, _, task, _, _ = run_live_agent.build_live_run(
        "degraded-canary", "test-model", client=_fake_client(), provider="openai"
    )

    assert isinstance(task, AgentTask)
    assert task.task
    assert set(AgentTask.model_fields) == {"task"}


def test_live_helper_passes_model_and_existing_tools_to_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def __init__(self, model: str, client: object = None) -> None:
            captured["model"] = model
            captured["client"] = client

    class FakeRunner:
        def __init__(self, model: object, observations: object, mutations: object) -> None:
            captured["runner_model"] = model
            captured["observations"] = observations
            captured["mutations"] = mutations

    monkeypatch.setattr(run_live_agent, "OpenAIResponsesModel", FakeModel)
    monkeypatch.setattr(run_live_agent, "AgentRunner", FakeRunner)
    fake_client = object()

    scenario, environment, task, runner, observations = run_live_agent.build_live_run(
        "degraded-canary", "gpt-6-luna", client=fake_client, provider="openai"
    )

    assert scenario.id == "degraded-canary"
    assert isinstance(environment, RolloutEnvironment)
    assert task.task == scenario.task
    assert isinstance(runner, FakeRunner)
    assert captured["model"] == "gpt-6-luna"
    assert captured["client"] is fake_client
    assert captured["observations"] is observations
    assert isinstance(observations, RolloutObservationTools)
    assert isinstance(captured["mutations"], RolloutMutationTools)


def test_safe_summary_contains_runtime_result_without_private_truth() -> None:
    client = _fake_client()

    summary = run_live_agent.execute_live_run(
        "degraded-canary", "gpt-6-luna", client=client, provider="openai"
    )

    assert summary["scenario_id"] == "degraded-canary"
    assert summary["model_id"] == "gpt-6-luna"
    assert summary["final_answer"]["decision"] == "rollback"
    assert summary["tool_interactions"][0]["tool"] == "rollback_rollout"
    assert summary["tool_interactions"][0]["status"] == "success"
    assert summary["tool_interactions"][0]["result"]["phase"] == "RolledBack"
    assert summary["final_rollout_state"] == {
        "phase": "RolledBack",
        "candidate_weight": 0,
    }
    assert summary["mutation_count"] == 1

    summary_text = json.dumps(summary)
    for private_field in (
        "OPENAI_API_KEY",
        "expected_decision",
        "expected_final_state",
        "required_evidence",
        "max_mutations",
    ):
        assert private_field not in summary_text


def test_missing_api_key_stops_before_provider_construction(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    def fail_if_constructed(*args: object, **kwargs: object) -> None:
        pytest.fail("provider must not be constructed without configured credentials")

    monkeypatch.setattr(run_live_agent, "OpenAIResponsesModel", fail_if_constructed)

    exit_code = run_live_agent.main(["--provider", "openai", "--model", "gpt-6-luna"])

    assert exit_code == 2
    assert "LIVE RUN BLOCKED: OPENAI_API_KEY is not configured." in capsys.readouterr().err


def test_live_cli_accepts_explicit_groq_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    parsed = run_live_agent.build_parser().parse_args(
        ["--provider", "groq", "--model", "test-model"]
    )
    assert parsed.provider == "groq"
    assert parsed.model == "test-model"


def test_live_helper_selects_groq_agent_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class FakeGroqModel:
        def __init__(self, model: str, client: object = None) -> None:
            captured.update(model=model, client=client)

    monkeypatch.setattr(run_live_agent, "GroqResponsesModel", FakeGroqModel)
    client = object()

    _, _, _, runner, _ = run_live_agent.build_live_run(
        "degraded-canary", "caller-model", client=client, provider="groq"
    )

    assert isinstance(runner, run_live_agent.AgentRunner)
    assert captured == {"model": "caller-model", "client": client}


def test_live_cli_requires_provider_and_model() -> None:
    parser = run_live_agent.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--provider", "openai"])


def test_live_cli_rejects_invalid_provider() -> None:
    with pytest.raises(SystemExit):
        run_live_agent.build_parser().parse_args(["--provider", "other", "--model", "m"])


def test_fake_live_execution_makes_no_network_calls() -> None:
    client = _fake_client()

    run_live_agent.execute_live_run("degraded-canary", "test-model", client=client, provider="openai")

    assert len(client.responses.calls) == 2
