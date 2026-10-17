import hashlib
import sys
from pathlib import Path

import pytest

from rollouteval.agent.external import (
    ExternalAgentSpec,
    ExternalEnvironmentVariable,
    ExternalAgentMalformedJSONError,
    ExternalAgentNonZeroExitError,
    ExternalAgentStderrLimitError,
    ExternalAgentStdoutLimitError,
    ExternalAgentTimeoutError,
    InvalidExternalAgentResponseError,
    UnsupportedExternalAgentProtocolError,
    build_external_agent,
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
FIXTURE = ROOT / "tests" / "fixtures" / "external_agent.py"


def _agent(
    tmp_path,
    *,
    mode="valid",
    timeout=10,
    stdout=1_048_576,
    stderr=65_536,
    env=(),
    command=None,
):
    spec = ExternalAgentSpec(
        agent_id="external-fixture",
        version="1.0",
        implementation_sha256=hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        command=command or (sys.executable, str(FIXTURE)),
        working_directory=ROOT,
        environment=(ExternalEnvironmentVariable(name="FIXTURE_MODE", value=mode), *env),
        timeout_seconds=timeout,
        max_stdout_bytes=stdout,
        max_stderr_bytes=stderr,
    )
    return build_external_agent(spec)


def _benchmark(*ids):
    return BenchmarkSpec(
        benchmark_id="external-fixture-benchmark",
        version="1",
        tasks=tuple(
            load_scenario(ROOT / "scenarios" / f"{scenario_id}.yaml")
            for scenario_id in ids
        ),
    )


def _run(tmp_path, agent, benchmark, *, trials=1):
    runs = RunArtifactStore(tmp_path / "runs")
    benchmarks = BenchmarkArtifactStore(tmp_path / "benchmarks")
    result = BenchmarkRunner(runs, benchmarks).run(
        agent, benchmark, trials_per_task=trials
    )
    return result, runs, benchmarks


def test_external_agent_runs_through_benchmark_and_report_path(tmp_path):
    result, runs, benchmarks = _run(
        tmp_path, _agent(tmp_path), _benchmark("healthy-canary", "degraded-canary")
    )
    report = BenchmarkReportService(benchmarks, runs).get(result.benchmark_run_id)
    assert report.completed_trials == 2
    assert [trial.trial.scenario_id for trial in report.trials] == [
        "healthy-canary",
        "degraded-canary",
    ]
    assert all(trial.run_report is not None for trial in report.trials)
    assert report.agent.connection_type == "local_subprocess"
    assert report.agent.protocol_version == 1


def test_objectively_wrong_external_agent_is_evaluated_not_process_failure(tmp_path):
    result, runs, benchmarks = _run(
        tmp_path, _agent(tmp_path, mode="wrong"), _benchmark("healthy-canary")
    )
    report = BenchmarkReportService(benchmarks, runs).get(result.benchmark_run_id)
    assert report.completed_trials == 1
    assert report.trials[0].run_report is not None
    decision = next(
        item for item in report.trials[0].run_report.criteria
        if item.evaluator_name == "decision"
    )
    assert decision.status.value == "FAIL"


@pytest.mark.parametrize(
    ("mode", "error_type", "limits", "timeout"),
    [
        ("malformed", ExternalAgentMalformedJSONError.__name__, {}, 10),
        ("timeout", ExternalAgentTimeoutError.__name__, {}, 0.2),
        ("nonzero", ExternalAgentNonZeroExitError.__name__, {}, 10),
        ("stdout_oversize", ExternalAgentStdoutLimitError.__name__, {"stdout": 128}, 10),
        ("stderr_oversize", ExternalAgentStderrLimitError.__name__, {"stderr": 128}, 10),
        ("protocol_mismatch", UnsupportedExternalAgentProtocolError.__name__, {}, 10),
        ("invalid_schema", InvalidExternalAgentResponseError.__name__, {}, 10),
    ],
)
def test_external_protocol_failures_are_agent_errors_and_trials_continue(
    tmp_path, mode, error_type, limits, timeout
):
    result, runs, benchmarks = _run(
        tmp_path,
        _agent(tmp_path, mode=mode, timeout=timeout, **limits),
        _benchmark("healthy-canary", "degraded-canary"),
    )
    report = BenchmarkReportService(benchmarks, runs).get(result.benchmark_run_id)
    assert report.agent_execution_error_trials == 2
    assert all(item.trial.status is TrialStatus.AGENT_EXECUTION_ERROR for item in report.trials)
    assert all(item.trial.error_type == error_type for item in report.trials)
    assert all(item.run_report is None for item in report.trials)


def test_failed_trial_does_not_erase_or_block_later_valid_trial(tmp_path):
    result, runs, benchmarks = _run(
        tmp_path,
        _agent(tmp_path, mode="malformed_first"),
        _benchmark("healthy-canary", "degraded-canary"),
    )
    report = BenchmarkReportService(benchmarks, runs).get(result.benchmark_run_id)
    assert report.trials[0].trial.status is TrialStatus.AGENT_EXECUTION_ERROR
    assert report.trials[0].trial.error_type == ExternalAgentMalformedJSONError.__name__
    assert report.trials[0].run_report is None
    assert report.trials[1].trial.status is TrialStatus.COMPLETED
    assert report.trials[1].run_report is not None


def test_missing_executable_is_classified_as_agent_execution_error(tmp_path):
    result, runs, benchmarks = _run(
        tmp_path,
        _agent(tmp_path, command=("rollouteval-no-such-executable",)),
        _benchmark("healthy-canary"),
    )
    report = BenchmarkReportService(benchmarks, runs).get(result.benchmark_run_id)
    assert report.agent_execution_error_trials == 1
    assert report.trials[0].trial.error_type == "ExternalAgentExecutableNotFoundError"
    assert report.trials[0].run_report is None


def test_repeated_external_trials_have_distinct_runs_and_environment_is_allowlisted(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GROQ_API_KEY", "dummy-groq-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-secret")
    monkeypatch.setenv("SECRET_TOKEN", "dummy-secret-token")
    agent = _agent(
        tmp_path,
        mode="environment",
        env=(ExternalEnvironmentVariable(name="SAFE_AGENT_FLAG", value="allowed"),),
    )
    result, runs, benchmarks = _run(
        tmp_path, agent, _benchmark("healthy-canary"), trials=2
    )
    report = BenchmarkReportService(benchmarks, runs).get(result.benchmark_run_id)
    assert len({item.trial.trial_id for item in report.trials}) == 2
    assert len({item.trial.run_id for item in report.trials}) == 2
    for artifact in (tmp_path / "runs").glob("*.json"):
        assert "dummy-groq-secret" not in artifact.read_text()
        assert "dummy-openai-secret" not in artifact.read_text()
        assert "dummy-secret-token" not in artifact.read_text()
    for artifact in (tmp_path / "benchmarks").glob("*.json"):
        assert "dummy-groq-secret" not in artifact.read_text()
        assert "dummy-openai-secret" not in artifact.read_text()
        assert "dummy-secret-token" not in artifact.read_text()
    for item in report.trials:
        assert item.run_report is not None
        run = runs.load(item.trial.run_id)
        assert "SAFE_AGENT_FLAG" in run.agent_run.final_answer.explanation
        assert "GROQ_API_KEY" not in run.agent_run.final_answer.explanation
        assert "OPENAI_API_KEY" not in run.agent_run.final_answer.explanation
        assert "SECRET_TOKEN" not in run.agent_run.final_answer.explanation
