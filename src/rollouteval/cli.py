"""Command-line workflow for configuring agents and running benchmarks."""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError

from rollouteval.agent.external import ExternalAgentSpec, ExternalEnvironmentVariable, build_external_agent
from rollouteval.agent.remote import RemoteAgentSpec, build_remote_agent
from rollouteval.benchmark import (
    BenchmarkArtifactStore,
    BenchmarkReportService,
    BenchmarkRunner,
    BenchmarkSpec,
)
from rollouteval.cli_config import (
    AgentConfiguration,
    CLIConfigurationError,
    CLIConfigurationStore,
    default_artifact_dir,
    default_config_dir,
)
from rollouteval.run_artifacts import RunArtifactStore, RunReportService
from rollouteval.scenarios import load_scenario


def _add_json_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rollouteval", description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=default_config_dir())
    parser.add_argument("--artifact-dir", type=Path, default=default_artifact_dir())
    commands = parser.add_subparsers(dest="group", required=True)

    agent = commands.add_parser("agent", help="manage configured agent connections")
    agent_commands = agent.add_subparsers(dest="action", required=True)
    add_subprocess = agent_commands.add_parser("add-subprocess", help="configure a local argv agent")
    add_subprocess.add_argument("--id", required=True)
    add_subprocess.add_argument("--version", required=True)
    add_subprocess.add_argument("--implementation-sha256", required=True)
    add_subprocess.add_argument("--executable", required=True)
    add_subprocess.add_argument("--arg", action="append", default=[], help="one argv item; repeat as needed")
    add_subprocess.add_argument("--working-directory", type=Path)
    add_subprocess.add_argument("--protocol-version", type=int, choices=(1,), default=1)
    add_subprocess.add_argument("--timeout", type=float, default=30)
    add_subprocess.add_argument("--allow-env", action="append", default=[], metavar="NAME",
                                help="explicit child environment variable reference; repeatable")
    add_subprocess.add_argument("--max-request-bytes", type=int, default=1_048_576)
    add_subprocess.add_argument("--max-stdout-bytes", type=int, default=1_048_576)
    add_subprocess.add_argument("--max-stderr-bytes", type=int, default=65_536)
    add_subprocess.add_argument("--max-steps", type=int, default=8)
    _add_json_option(add_subprocess)

    add_remote = agent_commands.add_parser("add-remote", help="configure a remote HTTP agent")
    add_remote.add_argument("--id", required=True)
    add_remote.add_argument("--version", required=True)
    add_remote.add_argument("--implementation-sha256", required=True)
    add_remote.add_argument("--endpoint", required=True)
    add_remote.add_argument("--protocol-version", type=int, choices=(1,), default=1)
    add_remote.add_argument("--timeout", type=float, default=30)
    add_remote.add_argument("--max-request-bytes", type=int, default=1_048_576)
    add_remote.add_argument("--max-response-bytes", type=int, default=1_048_576)
    add_remote.add_argument("--token-env", metavar="NAME", help="environment variable name; value is runtime-only")
    _add_json_option(add_remote)

    for action in ("list", "show", "remove"):
        item = agent_commands.add_parser(action, help=f"{action} configured agents")
        if action == "show":
            item.add_argument("agent_id")
        if action == "remove":
            item.add_argument("agent_id")
        _add_json_option(item)

    benchmark = commands.add_parser("benchmark", help="manage and execute benchmarks")
    benchmark_commands = benchmark.add_subparsers(dest="action", required=True)
    add_benchmark = benchmark_commands.add_parser("add", help="import scenarios into a benchmark definition")
    add_benchmark.add_argument("--id", required=True)
    add_benchmark.add_argument("--version", required=True)
    add_benchmark.add_argument("--scenario", action="append", required=True, type=Path,
                               help="scenario YAML path; order is benchmark order")
    add_benchmark.add_argument("--name")
    add_benchmark.add_argument("--description")
    _add_json_option(add_benchmark)
    for action in ("list", "show"):
        item = benchmark_commands.add_parser(action, help=f"{action} benchmark definitions")
        if action == "show":
            item.add_argument("benchmark_id")
        _add_json_option(item)
    run_benchmark = benchmark_commands.add_parser("run", help="execute a benchmark against a configured agent")
    run_benchmark.add_argument("--agent", required=True)
    run_benchmark.add_argument("--benchmark", required=True)
    run_benchmark.add_argument("--trials-per-task", type=int, default=1)
    _add_json_option(run_benchmark)

    run = commands.add_parser("run", help="inspect persisted ObservableRun artifacts")
    run_commands = run.add_subparsers(dest="action", required=True)
    for action in ("list", "show"):
        item = run_commands.add_parser(action, help=f"{action} persisted runs")
        if action == "show":
            item.add_argument("run_id")
        _add_json_option(item)

    report = commands.add_parser("report", help="inspect persisted benchmark reports")
    report_commands = report.add_subparsers(dest="action", required=True)
    for action in ("list", "show"):
        item = report_commands.add_parser(action, help=f"{action} benchmark reports")
        if action == "show":
            item.add_argument("benchmark_run_id")
        _add_json_option(item)
    return parser


def _safe_agent_view(configuration: AgentConfiguration) -> dict[str, Any]:
    result: dict[str, Any] = {
        "agent_id": configuration.agent_id,
        "version": configuration.version,
        "implementation_sha256": configuration.implementation_sha256,
        "connection_type": configuration.connection_type,
        "protocol_version": configuration.protocol_version,
        "timeout_seconds": configuration.timeout_seconds,
    }
    if configuration.connection_type == "local_subprocess":
        result["command_argv"] = list(configuration.command or ())
        result["allowed_environment_names"] = list(configuration.allowed_environment_names)
    else:
        result["endpoint"] = configuration.endpoint_url
        result["token_environment_name"] = configuration.token_environment_name
    return result


def _benchmark_view(benchmark: BenchmarkSpec) -> dict[str, Any]:
    return {
        "benchmark_id": benchmark.benchmark_id,
        "version": benchmark.version,
        "digest": benchmark.digest,
        "name": benchmark.name,
        "description": benchmark.description,
        "tasks": [
            {"scenario_id": task.id, "name": task.name, "description": task.description, "task": task.task}
            for task in benchmark.tasks
        ],
    }


def _summary_view(result) -> dict[str, Any]:
    summary = result.summary
    return {
        "benchmark_run_id": result.benchmark_run_id,
        "benchmark_id": result.benchmark.benchmark_id,
        "benchmark_version": result.benchmark.version,
        "benchmark_digest": result.benchmark_digest,
        "agent_id": result.agent.agent_id,
        "agent_version": result.agent.version,
        "trials_attempted": summary.trials_attempted,
        "tasks_attempted": summary.tasks_attempted,
        "completed_trials": summary.completed_trials,
        "agent_execution_error_trials": summary.agent_execution_error_trials,
        "provider_runtime_error_trials": summary.provider_runtime_error_trials,
        "infrastructure_error_trials": summary.infrastructure_error_trials,
        "criterion_counts": [item.model_dump(mode="json") for item in summary.criterion_counts],
        "artifact_reference": result.benchmark_run_id,
    }


def _emit(value: Any, *, json_mode: bool, human: str) -> None:
    if json_mode:
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(human)


def _resolve_agent(configuration: AgentConfiguration):
    if configuration.connection_type == "local_subprocess":
        missing = [name for name in configuration.allowed_environment_names if name not in os.environ]
        if missing:
            raise CLIConfigurationError(
                "required child environment variable is missing: " + ", ".join(missing)
            )
        spec = ExternalAgentSpec(
            agent_id=configuration.agent_id,
            version=configuration.version,
            implementation_sha256=configuration.implementation_sha256,
            command=configuration.command or (),
            working_directory=configuration.working_directory,
            environment=tuple(
                ExternalEnvironmentVariable(name=name, value=os.environ[name])
                for name in configuration.allowed_environment_names
            ),
            timeout_seconds=configuration.timeout_seconds,
            max_request_bytes=configuration.max_request_bytes,
            max_stdout_bytes=configuration.max_stdout_bytes,
            max_stderr_bytes=configuration.max_stderr_bytes,
            max_steps=configuration.max_steps,
            protocol_version=configuration.protocol_version,
        )
        return build_external_agent(spec)
    secret = None
    if configuration.token_environment_name:
        secret = os.environ.get(configuration.token_environment_name)
        if not secret:
            raise CLIConfigurationError(
                "required runtime token environment variable is missing: "
                + configuration.token_environment_name
            )
    spec = RemoteAgentSpec(
        agent_id=configuration.agent_id,
        version=configuration.version,
        implementation_sha256=configuration.implementation_sha256,
        endpoint_url=configuration.endpoint_url or "",
        timeout_seconds=configuration.timeout_seconds,
        max_request_bytes=configuration.max_request_bytes,
        max_response_bytes=configuration.max_response_bytes,
        protocol_version=configuration.protocol_version,
    )
    return build_remote_agent(spec, bearer_token=secret)


def _dispatch(args: argparse.Namespace) -> tuple[Any, str, int]:
    store = CLIConfigurationStore(args.config_dir)
    if args.group == "agent":
        if args.action == "add-subprocess":
            config = AgentConfiguration(
                agent_id=args.id, version=args.version,
                implementation_sha256=args.implementation_sha256,
                connection_type="local_subprocess", command=(args.executable, *args.arg),
                working_directory=str(args.working_directory) if args.working_directory else None,
                allowed_environment_names=tuple(args.allow_env),
                timeout_seconds=args.timeout, protocol_version=args.protocol_version,
                max_request_bytes=args.max_request_bytes, max_stdout_bytes=args.max_stdout_bytes,
                max_stderr_bytes=args.max_stderr_bytes, max_steps=args.max_steps,
            )
            path = store.save_agent(config)
            value = _safe_agent_view(config)
            value["configuration_path"] = str(path)
            return value, f"Saved subprocess agent {config.agent_id} ({config.version}).", 0
        if args.action == "add-remote":
            parts = urlsplit(args.endpoint)
            try:
                port = parts.port
            except ValueError:
                raise CLIConfigurationError("endpoint URL is malformed") from None
            if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username
                    or parts.password or parts.query or parts.fragment
                    or (port is not None and not 1 <= port <= 65535)
                    or any(ord(character) < 32 or character.isspace() for character in args.endpoint)):
                raise CLIConfigurationError("endpoint must be HTTP(S), credential-free, and have no query/fragment")
            config = AgentConfiguration(
                agent_id=args.id, version=args.version,
                implementation_sha256=args.implementation_sha256,
                connection_type="remote_http", endpoint_url=args.endpoint,
                timeout_seconds=args.timeout, protocol_version=args.protocol_version,
                max_request_bytes=args.max_request_bytes, max_response_bytes=args.max_response_bytes,
                token_environment_name=args.token_env,
            )
            path = store.save_agent(config)
            value = _safe_agent_view(config)
            value["configuration_path"] = str(path)
            return value, f"Saved remote agent {config.agent_id} ({config.version}).", 0
        if args.action == "list":
            values = [_safe_agent_view(item) for item in store.list_agents()]
            return values, "\n".join(
                f"{item['agent_id']}\t{item['version']}\t{item['connection_type']}\t{item['implementation_sha256'][:12]}"
                for item in values
            ) or "No configured agents.", 0
        if args.action == "show":
            config = store.load_agent(args.agent_id)
            return _safe_agent_view(config), json.dumps(_safe_agent_view(config), indent=2, sort_keys=True), 0
        store.remove_agent(args.agent_id)
        return {"removed": args.agent_id}, f"Removed local configuration for {args.agent_id}.", 0

    if args.group == "benchmark":
        if args.action == "add":
            try:
                tasks = tuple(load_scenario(path) for path in args.scenario)
            except Exception:
                raise CLIConfigurationError("scenario file could not be loaded or validated") from None
            benchmark = BenchmarkSpec(
                benchmark_id=args.id, version=args.version, tasks=tasks,
                name=args.name, description=args.description,
            )
            path = store.save_benchmark(benchmark)
            value = _benchmark_view(benchmark)
            value["configuration_path"] = str(path)
            return value, f"Saved benchmark {benchmark.benchmark_id} ({benchmark.version}), {len(tasks)} ordered tasks.", 0
        if args.action == "list":
            values = [_benchmark_view(item) for item in store.list_benchmarks()]
            return values, "\n".join(
                f"{item['benchmark_id']}\t{item['version']}\t{len(item['tasks'])} tasks\t{item['digest'][:12]}"
                for item in values
            ) or "No configured benchmarks.", 0
        if args.action == "show":
            benchmark = store.load_benchmark(args.benchmark_id)
            view = _benchmark_view(benchmark)
            return view, json.dumps(view, indent=2, sort_keys=True), 0
        configuration = store.load_agent(args.agent)
        benchmark = store.load_benchmark(args.benchmark)
        if args.trials_per_task < 1:
            raise CLIConfigurationError("trials-per-task must be at least one")
        agent_definition = _resolve_agent(configuration)
        run_root = args.artifact_dir / "runs"
        benchmark_root = args.artifact_dir / "benchmarks"
        run_store = RunArtifactStore(run_root)
        benchmark_store = BenchmarkArtifactStore(benchmark_root)
        result = BenchmarkRunner(run_store, benchmark_store).run(
            agent_definition, benchmark, trials_per_task=args.trials_per_task
        )
        value = _summary_view(result)
        human = (
            f"Benchmark run {result.benchmark_run_id}\n"
            f"Benchmark: {benchmark.benchmark_id} v{benchmark.version} ({benchmark.digest})\n"
            f"Agent: {result.agent.agent_id} v{result.agent.version}\n"
            f"Trials: {result.summary.trials_attempted} attempted, {result.summary.completed_trials} completed, "
            f"{result.summary.agent_execution_error_trials} agent errors, {result.summary.infrastructure_error_trials} infrastructure errors\n"
            + "\n".join(
                f"{item.evaluator_name} [{item.criterion_id or 'no criterion id'}]: PASS={item.passed} FAIL={item.failed} INCONCLUSIVE={item.inconclusive}"
                for item in result.summary.criterion_counts
            )
            + f"\nArtifact: {benchmark_root}"
        )
        exit_code = 4 if result.summary.infrastructure_error_trials else 0
        return value, human, exit_code

    if args.group == "run":
        reports = RunReportService(RunArtifactStore(args.artifact_dir / "runs"))
        if args.action == "list":
            values = [item.model_dump(mode="json") for item in reports.list()]
            human = "\n".join(f"{item['run_id']}\t{item['scenario_id']}\t{item['task']}" for item in values) or "No persisted runs."
            return values, human, 0
        report = reports.get(args.run_id)
        value = report.model_dump(mode="json")
        human = f"Run {report.run_id}\nScenario: {report.scenario_id}\nTask: {report.task}\n" + "\n".join(
            f"{item.evaluator_name} v{item.evaluator_version} [{item.criterion_id or 'no criterion id'}]: {item.status.value}\n"
            f"  {item.rationale}\n"
            f"  Evidence: {', '.join(item.evidence_references) or 'none'}\n"
            f"  Judge: {item.provider or 'n/a'} / {item.model or 'n/a'}; prompt SHA-256: {item.prompt_sha256 or 'n/a'}"
            for item in report.criteria
        )
        return value, human, 0

    reports = BenchmarkReportService(
        BenchmarkArtifactStore(args.artifact_dir / "benchmarks"),
        RunArtifactStore(args.artifact_dir / "runs"),
    )
    if args.action == "list":
        values = [item.model_dump(mode="json") for item in reports.list()]
        human = "\n".join(
            f"{item['benchmark_run_id']}\t{item['benchmark_id']} v{item['benchmark_version']}\t"
            f"{item['completed_trials']}/{item['trials_attempted']} completed"
            for item in values
        ) or "No persisted benchmark reports."
        return values, human, 0
    report = reports.get(args.benchmark_run_id)
    value = report.model_dump(mode="json")
    lines = [
        f"Benchmark run {report.benchmark_run_id}",
        f"Benchmark: {report.benchmark_id} v{report.benchmark_version} ({report.benchmark_digest})",
        f"Agent: {report.agent.agent_id} v{report.agent.version}",
        f"Trials: {report.trials_attempted} attempted, {report.completed_trials} completed, "
        f"{report.agent_execution_error_trials} agent errors, {report.infrastructure_error_trials} infrastructure errors",
    ]
    for item in report.trials:
        lines.append(f"Trial {item.trial.trial_id}: {item.trial.status.value}; run={item.trial.run_id or 'none'}")
        if item.trial.error_type:
            lines.append(f"  Error: {item.trial.error_type}")
        if item.run_report:
            for criterion in item.run_report.criteria:
                lines.append(
                    f"  {criterion.evaluator_name} [{criterion.criterion_id or 'no criterion id'}]: "
                    f"{criterion.status.value} — {criterion.rationale}"
                )
    return value, "\n".join(lines), 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        value, human, code = _dispatch(args)
        _emit(value, json_mode=args.json, human=human)
        return code
    except SystemExit as exc:
        return int(exc.code)
    except FileNotFoundError as exc:
        message = str(exc) if str(exc) in {"configured agent does not exist"} else "requested configuration or artifact was not found"
        print(f"rollouteval: {message}", file=sys.stderr)
        return 3
    except (CLIConfigurationError, ValidationError, ValueError) as exc:
        message = str(exc) if isinstance(exc, CLIConfigurationError) else "configuration or command data is invalid"
        print(f"rollouteval: {message}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"rollouteval: operation failed ({type(exc).__name__})", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
