import hashlib
import io
import json
import socket
import ssl
import sys
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

from rollouteval.agent.external import (
    ExternalAgentMalformedJSONError,
    InvalidExternalAgentResponseError,
    ExternalAgentSpec,
    UnsupportedExternalAgentProtocolError,
    build_external_agent,
)
from rollouteval.agent.remote import (
    RemoteAgentConfigurationError,
    RemoteAgentConnectTimeoutError,
    RemoteAgentConnectionError,
    RemoteAgentContentTypeError,
    RemoteAgentHTTPStatusError,
    RemoteAgentInvalidUTF8Error,
    RemoteAgentReadTimeoutError,
    RemoteAgentResponseLimitError,
    RemoteAgentSpec,
    RemoteAgentTLSFailureError,
    build_remote_agent,
)
from rollouteval.benchmark import (
    BenchmarkArtifactStore,
    BenchmarkReportService,
    BenchmarkRunner,
    BenchmarkSpec,
    TrialStatus,
)
from rollouteval.run_artifacts import RunArtifactStore
from rollouteval.scenarios import load_scenario

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_SHA256 = hashlib.sha256(b"fixture implementation version 1").hexdigest()
BEARER_SECRET = "dummy-remote-agent-token"


class _Response:
    def __init__(self, body, *, mode="valid", status=200, content_type="application/json"):
        self.status = status
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        if mode != "oversized":
            self.headers["Content-Length"] = str(len(body))
        self._body = body
        self._mode = mode
        self.read_sizes = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def getcode(self):
        return self.status

    def read(self, size):
        self.read_sizes.append(size)
        if self._mode == "read_timeout":
            raise socket.timeout("read timed out")
        return self._body[:size]


class _TransportState:
    def __init__(self, mode):
        self.mode = mode
        self.requests = []
        self.authorization = []
        self.paths = []
        self.responses = []
        self.read_sizes = []
        self.calls = 0


def _protocol_response(request, mode):
    scenario_id = request["scenario_id"]
    called = {item["call"]["tool"] for item in request["interactions"]}
    if scenario_id == "healthy-canary":
        required = ("get_rollout_state", "get_service_metrics")
        decision = "rollback" if mode == "wrong" else "promote"
    elif scenario_id == "degraded-canary":
        required = ("get_rollout_state", "get_service_metrics", "get_recent_events")
        decision = "rollback"
    else:
        required = ("get_rollout_state", "get_service_metrics")
        decision = "pause"
    missing = next((tool for tool in required if tool not in called), None)
    action = {
        "promote": "promote_rollout",
        "pause": "pause_rollout",
        "rollback": "rollback_rollout",
    }[decision]
    if missing is not None:
        output = {"kind": "tool_call", "tool": missing, "arguments": {}}
    elif action not in called:
        output = {"kind": "tool_call", "tool": action, "arguments": {}}
    else:
        output = {
            "kind": "final",
            "decision": decision,
            "explanation": "The observed rollout evidence supports this action.",
        }
    return {"protocol_version": 1, "response": {"output": output}}


@pytest.fixture
def transport_factory(monkeypatch):
    from rollouteval.agent import remote

    states = []

    def start(mode="valid"):
        state = _TransportState(mode)
        states.append(state)
        endpoint = f"https://remote-fixture.test/{len(states)}/respond"

        def fake_open(request, *, timeout):
            state.calls += 1
            state.paths.append(request.full_url)
            state.authorization.append(request.get_header("Authorization"))
            body = json.loads(request.data)
            state.requests.append(body)
            current_mode = state.mode
            if current_mode == "fail_first" and state.calls == 1:
                current_mode = "http_error"
            if current_mode == "connect_timeout":
                raise urllib.error.URLError(socket.timeout("connect timeout detail"))
            if current_mode == "connection_error":
                raise urllib.error.URLError(ConnectionRefusedError("connection detail"))
            if current_mode == "tls_error":
                raise urllib.error.URLError(ssl.SSLCertVerificationError("TLS detail"))
            if current_mode in {"http_error", "redirect"}:
                status = 302 if current_mode == "redirect" else 503
                raise urllib.error.HTTPError(
                    request.full_url,
                    status,
                    "private status detail",
                    Message(),
                    io.BytesIO(BEARER_SECRET.encode()),
                )
            if current_mode == "malformed":
                payload = b"not-json"
            elif current_mode == "invalid_utf8":
                payload = b"\xff"
            elif current_mode == "protocol_mismatch":
                payload = b'{"protocol_version":999,"response":{}}'
            elif current_mode == "invalid_schema":
                payload = b'{"protocol_version":1,"response":{"output":{}}}'
            elif current_mode == "oversized":
                payload = b"x" * 2048
            else:
                payload = json.dumps(_protocol_response(body, current_mode)).encode()
            content_type = "text/plain" if current_mode == "wrong_content_type" else "application/json"
            response = _Response(payload, mode=current_mode, content_type=content_type)
            state.responses.append(response)
            state.read_sizes.extend(response.read_sizes)
            return response

        monkeypatch.setattr(remote, "_open_remote_request", fake_open)
        return endpoint, state

    return start


def _agent(endpoint, *, timeout=5, max_response=1_048_576, token=BEARER_SECRET):
    spec = RemoteAgentSpec(
        agent_id="remote-fixture-agent",
        version="2026.1",
        implementation_sha256=IMPLEMENTATION_SHA256,
        endpoint_url=endpoint,
        timeout_seconds=timeout,
        max_response_bytes=max_response,
    )
    return build_remote_agent(spec, bearer_token=token)


def _benchmark(*scenario_ids):
    return BenchmarkSpec(
        benchmark_id="remote-agent-benchmark",
        version="1",
        tasks=tuple(
            load_scenario(ROOT / "scenarios" / f"{scenario_id}.yaml")
            for scenario_id in scenario_ids
        ),
    )


def _run(tmp_path, agent, benchmark, *, trials=1):
    runs = RunArtifactStore(tmp_path / "runs")
    benchmarks = BenchmarkArtifactStore(tmp_path / "benchmarks")
    result = BenchmarkRunner(runs, benchmarks).run(
        agent, benchmark, trials_per_task=trials
    )
    return result, runs, benchmarks


def test_valid_remote_agent_uses_benchmark_artifacts_and_report_path(
    tmp_path, transport_factory
):
    endpoint, state = transport_factory()
    agent = _agent(endpoint)
    result, runs, benchmarks = _run(
        tmp_path, agent, _benchmark("healthy-canary", "degraded-canary")
    )
    report = BenchmarkReportService(benchmarks, runs).get(result.benchmark_run_id)
    assert report.completed_trials == 2
    assert all(item.run_report is not None for item in report.trials)
    assert report.agent.connection_type == "remote_http"
    assert report.agent.protocol_version == 1
    assert state.authorization and set(state.authorization) == {f"Bearer {BEARER_SECRET}"}
    assert all("expected_decision" not in request for request in state.requests)
    assert all("expected_decision" not in request["task"] for request in state.requests)


def test_objectively_wrong_remote_output_remains_objective_fail(tmp_path, transport_factory):
    endpoint, _ = transport_factory("wrong")
    result, runs, benchmarks = _run(
        tmp_path, _agent(endpoint), _benchmark("healthy-canary")
    )
    report = BenchmarkReportService(benchmarks, runs).get(result.benchmark_run_id)
    assert report.completed_trials == 1
    decision = next(
        item for item in report.trials[0].run_report.criteria
        if item.evaluator_name == "decision"
    )
    assert decision.status.value == "FAIL"


@pytest.mark.parametrize(
    ("mode", "expected_error", "kwargs"),
    [
        ("http_error", RemoteAgentHTTPStatusError, {}),
        ("malformed", ExternalAgentMalformedJSONError, {}),
        ("invalid_utf8", RemoteAgentInvalidUTF8Error, {}),
        ("protocol_mismatch", UnsupportedExternalAgentProtocolError, {}),
        ("invalid_schema", InvalidExternalAgentResponseError, {}),
        ("oversized", RemoteAgentResponseLimitError, {"max_response": 128}),
        ("wrong_content_type", RemoteAgentContentTypeError, {}),
        ("redirect", RemoteAgentHTTPStatusError, {}),
    ],
)
def test_remote_response_failures_are_execution_errors_without_run(
    tmp_path, transport_factory, mode, expected_error, kwargs
):
    endpoint, state = transport_factory(mode)
    result, runs, benchmarks = _run(
        tmp_path, _agent(endpoint, **kwargs), _benchmark("healthy-canary")
    )
    report = BenchmarkReportService(benchmarks, runs).get(result.benchmark_run_id)
    trial = report.trials[0].trial
    assert trial.status is TrialStatus.AGENT_EXECUTION_ERROR
    assert report.trials[0].run_report is None
    if expected_error is not None:
        assert trial.error_type == expected_error.__name__
    if mode == "oversized":
        assert state.responses[0].read_sizes == [129]
    if mode == "redirect":
        assert state.calls == 1


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("connect_timeout", RemoteAgentConnectTimeoutError),
        ("connection_error", RemoteAgentConnectionError),
        ("tls_error", RemoteAgentTLSFailureError),
        ("read_timeout", RemoteAgentReadTimeoutError),
    ],
)
def test_network_failures_map_to_safe_agent_errors(
    tmp_path, transport_factory, mode, expected
):
    endpoint, _ = transport_factory(mode)
    timeout = 0.05 if "timeout" in mode else 5
    result, runs, benchmarks = _run(
        tmp_path, _agent(endpoint, timeout=timeout), _benchmark("healthy-canary")
    )
    report = BenchmarkReportService(benchmarks, runs).get(result.benchmark_run_id)
    assert report.trials[0].trial.status is TrialStatus.AGENT_EXECUTION_ERROR
    assert report.trials[0].trial.error_type == expected.__name__
    assert report.trials[0].run_report is None


def test_failed_remote_trial_continues_to_later_repeated_trial(tmp_path, transport_factory):
    endpoint, state = transport_factory("fail_first")
    result, runs, benchmarks = _run(
        tmp_path,
        _agent(endpoint),
        _benchmark("healthy-canary"),
        trials=2,
    )
    report = BenchmarkReportService(benchmarks, runs).get(result.benchmark_run_id)
    assert report.trials[0].trial.status is TrialStatus.AGENT_EXECUTION_ERROR
    assert report.trials[0].run_report is None
    assert report.trials[1].trial.status is TrialStatus.COMPLETED
    assert report.trials[1].run_report is not None
    assert report.trials[0].trial.trial_id != report.trials[1].trial.trial_id
    assert report.trials[0].trial.run_id is None
    assert report.trials[1].trial.run_id is not None
    assert len({request["run_id"] for request in state.requests}) == 2


def test_remote_secret_is_runtime_only_and_not_in_artifacts_or_errors(
    tmp_path, transport_factory
):
    endpoint, _ = transport_factory("http_error")
    agent = _agent(endpoint)
    assert agent.identity.connection_type == "remote_http"
    assert BEARER_SECRET not in repr(agent.identity)
    assert BEARER_SECRET not in repr(agent.executor)
    assert BEARER_SECRET not in str(RemoteAgentHTTPStatusError())
    result, runs, benchmarks = _run(tmp_path, agent, _benchmark("healthy-canary"))
    report = BenchmarkReportService(benchmarks, runs).get(result.benchmark_run_id)
    assert BEARER_SECRET not in repr(report)
    for directory in (tmp_path / "runs", tmp_path / "benchmarks"):
        for artifact in directory.glob("*.json"):
            assert BEARER_SECRET not in artifact.read_text()


@pytest.mark.parametrize(
    "endpoint",
    [
        "file:///tmp/agent",
        "ftp://localhost/agent",
        "http://user:pass@localhost/agent",
        "http://[bad",
    ],
)
def test_invalid_remote_endpoint_is_agent_execution_error(tmp_path, transport_factory, endpoint):
    agent = _agent(endpoint)
    result, runs, benchmarks = _run(
        tmp_path, agent, _benchmark("healthy-canary")
    )
    report = BenchmarkReportService(benchmarks, runs).get(result.benchmark_run_id)
    assert report.trials[0].trial.status is TrialStatus.AGENT_EXECUTION_ERROR
    assert report.trials[0].trial.error_type == RemoteAgentConfigurationError.__name__
    assert report.trials[0].run_report is None


def test_remote_and_subprocess_transports_preserve_equivalent_agent_semantics(
    tmp_path, transport_factory
):
    endpoint, _ = transport_factory()
    remote_result, remote_runs, remote_benchmarks = _run(
        tmp_path / "remote", _agent(endpoint, token=None), _benchmark("healthy-canary")
    )
    subprocess_spec = ExternalAgentSpec(
        agent_id="subprocess-fixture",
        version="1",
        implementation_sha256=hashlib.sha256(
            (ROOT / "tests" / "fixtures" / "external_agent.py").read_bytes()
        ).hexdigest(),
        command=(sys.executable, str(ROOT / "tests" / "fixtures" / "external_agent.py")),
        working_directory=ROOT,
    )
    local_result, local_runs, local_benchmarks = _run(
        tmp_path / "local", build_external_agent(subprocess_spec), _benchmark("healthy-canary")
    )
    remote_report = BenchmarkReportService(remote_benchmarks, remote_runs).get(
        remote_result.benchmark_run_id
    )
    local_report = BenchmarkReportService(local_benchmarks, local_runs).get(
        local_result.benchmark_run_id
    )
    remote_run = remote_runs.load(remote_report.trials[0].trial.run_id)
    local_run = local_runs.load(local_report.trials[0].trial.run_id)
    assert remote_run.agent_run.final_answer == local_run.agent_run.final_answer
    assert [item.status for item in remote_report.trials[0].run_report.criteria] == [
        item.status for item in local_report.trials[0].run_report.criteria
    ]


def test_redirect_handler_refuses_redirect_targets():
    from rollouteval.agent.remote import _NoRedirectHandler

    handler = _NoRedirectHandler()
    assert handler.redirect_request(
        None, None, 302, "Found", {}, "https://other.invalid"
    ) is None
