import hashlib
import json
import os
import sys
from pathlib import Path

from rollouteval.cli import main


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "external_agent.py"
DIGEST = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()


def _base(tmp_path):
    return ["--config-dir", str(tmp_path / "config"), "--artifact-dir", str(tmp_path / "data")]


def _configure_benchmark(tmp_path, capsys, *scenario_ids):
    argv = _base(tmp_path) + ["benchmark", "add", "--id", "cli-test", "--version", "1"]
    for scenario_id in scenario_ids:
        argv.extend(["--scenario", str(ROOT / "scenarios" / f"{scenario_id}.yaml")])
    assert main(argv) == 0
    capsys.readouterr()


def _configure_agent(tmp_path, capsys, *, mode="valid"):
    env_name = "FIXTURE_MODE"
    os.environ[env_name] = mode
    argv = _base(tmp_path) + [
        "agent", "add-subprocess", "--id", "candidate", "--version", "1.0",
        "--implementation-sha256", DIGEST, "--executable", sys.executable,
        "--arg", str(FIXTURE), "--working-directory", str(ROOT), "--allow-env", env_name,
    ]
    assert main(argv) == 0
    capsys.readouterr()
    return env_name


def test_agent_config_persists_safe_argv_and_only_environment_references(tmp_path, capsys):
    secret = "dummy-runtime-secret-928374"
    os.environ["ROLL_OUT_TEST_SECRET"] = secret
    argv = _base(tmp_path) + [
        "agent", "add-subprocess", "--id", "safe-agent", "--version", "1",
        "--implementation-sha256", DIGEST, "--executable", sys.executable,
        "--arg", str(FIXTURE), "--allow-env", "ROLL_OUT_TEST_SECRET",
    ]
    assert main(argv) == 0
    capsys.readouterr()
    saved = b"".join(path.read_bytes() for path in (tmp_path / "config").rglob("*.json"))
    assert secret.encode() not in saved
    assert b"ROLL_OUT_TEST_SECRET" in saved
    assert main(_base(tmp_path) + ["agent", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["connection_type"] == "local_subprocess"


def test_remote_agent_config_saves_token_reference_but_not_value(tmp_path, capsys):
    token = "dummy-cli-token-938475"
    os.environ["ROLL_OUT_REMOTE_TOKEN"] = token
    argv = _base(tmp_path) + [
        "agent", "add-remote", "--id", "remote", "--version", "v1",
        "--implementation-sha256", "a" * 64, "--endpoint", "https://agent.invalid/respond",
        "--token-env", "ROLL_OUT_REMOTE_TOKEN",
    ]
    assert main(argv) == 0
    capsys.readouterr()
    saved = b"".join(path.read_bytes() for path in (tmp_path / "config").rglob("*.json"))
    assert token.encode() not in saved
    assert b"ROLL_OUT_REMOTE_TOKEN" in saved


def test_remote_endpoint_credentials_and_query_are_rejected(tmp_path, capsys):
    argv = _base(tmp_path) + [
        "agent", "add-remote", "--id", "remote", "--version", "v1",
        "--implementation-sha256", "a" * 64, "--endpoint", "https://user:pass@example.test/a?token=x",
    ]
    assert main(argv) == 2
    assert "endpoint" in capsys.readouterr().err
    malformed = _base(tmp_path) + [
        "agent", "add-remote", "--id", "malformed", "--version", "v1",
        "--implementation-sha256", "a" * 64, "--endpoint", "https://agent.invalid:not-a-port/respond",
    ]
    assert main(malformed) == 2
    assert "endpoint" in capsys.readouterr().err


def test_missing_runtime_secret_stops_before_benchmark_execution(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ABSENT_AGENT_TOKEN", raising=False)
    argv = _base(tmp_path) + [
        "agent", "add-remote", "--id", "remote", "--version", "v1",
        "--implementation-sha256", "a" * 64, "--endpoint", "https://agent.invalid/respond",
        "--token-env", "ABSENT_AGENT_TOKEN",
    ]
    assert main(argv) == 0
    capsys.readouterr()
    _configure_benchmark(tmp_path, capsys, "healthy-canary")
    assert main(_base(tmp_path) + ["benchmark", "run", "--agent", "remote", "--benchmark", "cli-test"]) == 2
    assert "ABSENT_AGENT_TOKEN" in capsys.readouterr().err
    assert not (tmp_path / "data").exists()


def test_benchmark_add_list_show_and_digest(tmp_path, capsys):
    _configure_benchmark(tmp_path, capsys, "healthy-canary", "degraded-canary")
    assert main(_base(tmp_path) + ["benchmark", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["tasks"][0]["scenario_id"] == "healthy-canary"
    assert listed[0]["tasks"][1]["scenario_id"] == "degraded-canary"
    assert len(listed[0]["digest"]) == 64
    assert main(_base(tmp_path) + ["benchmark", "show", "cli-test", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["benchmark_id"] == "cli-test"


def test_cli_smoke_configure_run_reopen_report_and_trial_run(tmp_path, capsys):
    _configure_benchmark(tmp_path, capsys, "healthy-canary", "degraded-canary")
    _configure_agent(tmp_path, capsys)
    assert main(_base(tmp_path) + ["agent", "show", "candidate", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["implementation_sha256"] == DIGEST
    assert main(_base(tmp_path) + ["benchmark", "show", "cli-test"]) == 0
    capsys.readouterr()

    assert main(_base(tmp_path) + ["benchmark", "run", "--agent", "candidate", "--benchmark", "cli-test", "--trials-per-task", "1", "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["trials_attempted"] == summary["completed_trials"] == 2
    benchmark_run_id = summary["benchmark_run_id"]

    assert main(_base(tmp_path) + ["report", "list", "--json"]) == 0
    reports = json.loads(capsys.readouterr().out)
    assert len(reports) == 1
    assert main(_base(tmp_path) + ["report", "show", benchmark_run_id, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report["trials"]) == 2
    run_id = report["trials"][0]["trial"]["run_id"]
    assert main(_base(tmp_path) + ["run", "list", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2
    assert main(_base(tmp_path) + ["run", "show", run_id, "--json"]) == 0
    run_report = json.loads(capsys.readouterr().out)
    assert run_report["run_id"] == run_id
    assert run_report["criteria"]


def test_objective_fail_is_reported_without_cli_error(tmp_path, capsys):
    _configure_benchmark(tmp_path, capsys, "healthy-canary")
    _configure_agent(tmp_path, capsys, mode="wrong")
    assert main(_base(tmp_path) + ["benchmark", "run", "--agent", "candidate", "--benchmark", "cli-test"]) == 0
    summary = capsys.readouterr().out
    benchmark_run_id = summary.splitlines()[0].split()[-1]
    assert main(_base(tmp_path) + ["report", "show", benchmark_run_id, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    objective = next(
        criterion
        for criterion in report["trials"][0]["run_report"]["criteria"]
        if criterion["evaluator_name"] == "decision"
    )
    assert objective["status"] == "FAIL"


def test_agent_execution_error_is_persisted_and_reported(tmp_path, capsys):
    _configure_benchmark(tmp_path, capsys, "healthy-canary", "degraded-canary")
    _configure_agent(tmp_path, capsys, mode="malformed_first")
    assert main(_base(tmp_path) + ["benchmark", "run", "--agent", "candidate", "--benchmark", "cli-test", "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["agent_execution_error_trials"] == 1
    assert summary["completed_trials"] == 1
    assert main(_base(tmp_path) + ["report", "show", summary["benchmark_run_id"], "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["trials"][0]["trial"]["status"] == "AGENT_EXECUTION_ERROR"
    assert report["trials"][1]["run_report"] is not None


def test_corrupt_agent_config_and_corrupt_recognized_artifact_fail_safely(tmp_path, capsys):
    _configure_agent(tmp_path, capsys)
    agent_path = next((tmp_path / "config" / "agents").glob("*.json"))
    agent_path.write_text("{broken", encoding="utf-8")
    assert main(_base(tmp_path) + ["agent", "list"]) == 2
    assert "malformed" in capsys.readouterr().err

    run_id = "corrupt-run"
    import hashlib

    filename = "run-" + hashlib.sha256(run_id.encode()).hexdigest() + ".json"
    run_dir = tmp_path / "data" / "runs"
    run_dir.mkdir(parents=True)
    (run_dir / filename).write_text("{bad", encoding="utf-8")
    assert main(_base(tmp_path) + ["run", "show", run_id]) == 4
    assert "RunArtifactCorruptError" in capsys.readouterr().err


def test_agent_remove_only_removes_configuration(tmp_path, capsys):
    _configure_agent(tmp_path, capsys)
    assert main(_base(tmp_path) + ["agent", "remove", "candidate"]) == 0
    assert "Removed" in capsys.readouterr().out
    assert not list((tmp_path / "config" / "agents").glob("*.json"))


def test_missing_agent_and_benchmark_use_not_found_exit_code(tmp_path, capsys):
    assert main(_base(tmp_path) + ["agent", "show", "missing"]) == 3
    assert "does not exist" in capsys.readouterr().err
    assert main(_base(tmp_path) + ["benchmark", "show", "missing"]) == 3
    assert "not found" in capsys.readouterr().err


def test_remote_agent_cli_workflow_uses_local_http_fixture(tmp_path, capsys, monkeypatch):
    observed = {}
    token = "dummy-local-http-token-8293"
    monkeypatch.setenv("CLI_REMOTE_TOKEN", token)

    def fake_post(spec, payload, bearer_token):
        observed["authorization"] = bearer_token.get_secret_value()
        request = json.loads(payload)
        observed["scenario_id"] = request["scenario_id"]
        return json.dumps({
            "protocol_version": 1,
            "response": {"output": {
                "kind": "final", "decision": "promote",
                "explanation": "The deterministic HTTP fixture returned a valid answer.",
            }},
        }).encode()

    monkeypatch.setattr("rollouteval.agent.remote._post_json", fake_post)
    argv = _base(tmp_path) + [
        "agent", "add-remote", "--id", "local-remote", "--version", "1",
        "--implementation-sha256", "b" * 64,
        "--endpoint", "http://agent.invalid/respond", "--token-env", "CLI_REMOTE_TOKEN",
    ]
    assert main(argv) == 0
    capsys.readouterr()
    _configure_benchmark(tmp_path, capsys, "healthy-canary")
    assert main(_base(tmp_path) + [
        "benchmark", "run", "--agent", "local-remote", "--benchmark", "cli-test", "--json",
    ]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["completed_trials"] == 1
    assert observed["authorization"] == token
    assert observed["scenario_id"] == "healthy-canary"

    assert main(_base(tmp_path) + ["report", "show", summary["benchmark_run_id"], "--json"]) == 0
    capsys.readouterr()
    serialized = b"".join(path.read_bytes() for path in (tmp_path / "config").rglob("*.json"))
    serialized += b"".join(path.read_bytes() for path in (tmp_path / "data").rglob("*.json"))
    assert token.encode() not in serialized
