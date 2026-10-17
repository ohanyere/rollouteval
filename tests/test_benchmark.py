from datetime import UTC, datetime
from pathlib import Path

import pytest

from rollouteval.agent import (
    AgentFinalAnswer,
    AgentModelResponse,
    AgentToolCall,
    AgentToolName,
)
from rollouteval.benchmark import (
    AgentDefinition,
    AgentExecutionError,
    AgentIdentity,
    BenchmarkArtifactConflictError,
    BenchmarkArtifactCorruptError,
    BenchmarkArtifactStore,
    BenchmarkArtifactWriteError,
    BenchmarkReportService,
    BenchmarkRunner,
    BenchmarkSpec,
    ModelAgentExecutor,
    ProviderRuntimeError,
    TrialStatus,
    UnsupportedBenchmarkArtifactVersionError,
)
from rollouteval.domain.scenario import Decision, Scenario
from rollouteval.evaluation import GraderStatus
from rollouteval.evaluation.groundedness import GroundednessJudgment
from rollouteval.run_artifacts import RunArtifactStore
from rollouteval.scenarios import load_scenario

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_IDS = ("healthy-canary", "degraded-canary", "insufficient-evidence")
EVIDENCE_TO_TOOL = {
    "rollout_state": AgentToolName.GET_ROLLOUT_STATE,
    "service_metrics": AgentToolName.GET_SERVICE_METRICS,
    "runtime_events": AgentToolName.GET_RECENT_EVENTS,
}
DECISION_TO_TOOL = {
    Decision.PROMOTE: AgentToolName.PROMOTE_ROLLOUT,
    Decision.PAUSE: AgentToolName.PAUSE_ROLLOUT,
    Decision.ROLLBACK: AgentToolName.ROLLBACK_ROLLOUT,
}


def _scenario(scenario_id: str) -> Scenario:
    scenario = load_scenario(ROOT / "scenarios" / f"{scenario_id}.yaml")
    return scenario.model_copy(update={"task": f"{scenario.id}: {scenario.task}"})


class ScriptedAgent:
    def __init__(self, scenarios, *, wrong_tasks=(), error_tasks=(), provider_error_tasks=()):
        self.by_task = {scenario.task: scenario for scenario in scenarios}
        self.wrong_tasks = set(wrong_tasks)
        self.error_tasks = set(error_tasks)
        self.provider_error_tasks = set(provider_error_tasks)
        self.calls = []

    def respond(self, task, interactions):
        scenario = self.by_task[task.task]
        self.calls.append(scenario.id)
        if scenario.id in self.error_tasks:
            raise AgentExecutionError("ScriptedAgentError")
        if scenario.id in self.provider_error_tasks:
            raise ProviderRuntimeError("scripted provider runtime error")
        requested = {interaction.call.tool for interaction in interactions}
        for evidence in scenario.required_evidence:
            tool = EVIDENCE_TO_TOOL[evidence]
            if tool not in requested:
                return AgentModelResponse(
                    output=AgentToolCall(kind="tool_call", tool=tool)
                )
        decision = scenario.expected_decision
        if scenario.id in self.wrong_tasks:
            decision = next(item for item in Decision if item is not decision)
        action = DECISION_TO_TOOL[decision]
        if action not in requested:
            return AgentModelResponse(
                output=AgentToolCall(kind="tool_call", tool=action)
            )
        return AgentModelResponse(
            output=AgentFinalAnswer(
                kind="final",
                decision=decision,
                explanation="The observed rollout evidence supports this action.",
            )
        )


class ScriptedJudge:
    provider_name = "test-provider"
    model = "test-judge"
    prompt_sha256 = "b" * 64

    def __init__(self, statuses=(), *, error=None):
        self.statuses = iter(statuses)
        self.error = error
        self.contexts = []

    def judge(self, context):
        self.contexts.append(context)
        if self.error is not None:
            raise self.error
        return GroundednessJudgment(
            status=next(self.statuses),
            reason="Scripted groundedness result.",
        )


def _benchmark(scenarios):
    return BenchmarkSpec(
        benchmark_id="canary-core",
        version="1.0.0",
        name="Canary core",
        tasks=tuple(scenarios),
    )


def _agent(model):
    return AgentDefinition(
        identity=AgentIdentity(
            agent_id="scripted-rollout-agent",
            version="2026.10",
            implementation_sha256="a" * 64,
        ),
        executor=ModelAgentExecutor(model),
    )


def _run(tmp_path, scenarios, model, *, trials_per_task=1, judge=None):
    runs = RunArtifactStore(tmp_path / "runs")
    benchmarks = BenchmarkArtifactStore(tmp_path / "benchmarks")
    result = BenchmarkRunner(runs, benchmarks).run(
        _agent(model),
        _benchmark(scenarios),
        trials_per_task=trials_per_task,
        groundedness_judge=judge,
    )
    return result, runs, benchmarks


def test_all_tasks_complete_and_reporting_preserves_benchmark_order(tmp_path):
    scenarios = [_scenario(item) for item in SCENARIO_IDS[:2]]
    model = ScriptedAgent(scenarios)
    result, runs, benchmarks = _run(tmp_path, scenarios, model)

    report = BenchmarkReportService(benchmarks, runs).get(result.benchmark_run_id)
    assert [trial.trial.scenario_id for trial in report.trials] == [
        scenario.id for scenario in scenarios
    ]
    assert report.tasks_attempted == report.trials_attempted == 2
    assert report.completed_trials == 2
    assert all(trial.run_report is not None for trial in report.trials)
    assert [trial.trial.run_id for trial in report.trials] == [
        trial.run_report.run_id for trial in report.trials
    ]
    assert not hasattr(report, "overall_status")


def test_mixed_objective_and_groundedness_counts_are_not_aggregated(tmp_path):
    scenarios = [_scenario(item) for item in SCENARIO_IDS[:2]]
    model = ScriptedAgent(scenarios, wrong_tasks={scenarios[1].id})
    judge = ScriptedJudge([GraderStatus.PASS, GraderStatus.FAIL])
    result, runs, benchmarks = _run(tmp_path, scenarios, model, judge=judge)

    report = BenchmarkReportService(benchmarks, runs).get(result.benchmark_run_id)
    decision = next(item for item in report.criterion_counts if item.evaluator_name == "decision")
    grounded = next(
        item for item in report.criterion_counts if item.criterion_id == "groundedness"
    )
    assert (decision.passed, decision.failed) == (1, 1)
    assert (grounded.passed, grounded.failed) == (1, 1)
    wrong_run = report.trials[1].run_report
    assert wrong_run is not None
    assert (
        next(
            item for item in wrong_run.criteria if item.evaluator_name == "decision"
        ).status
        is GraderStatus.FAIL
    )
    assert (
        next(
            item for item in wrong_run.criteria if item.criterion_id == "groundedness"
        ).status
        is GraderStatus.FAIL
    )
    assert not hasattr(report, "overall_status")


def test_failed_agent_trial_is_recorded_and_independent_tasks_continue(tmp_path):
    scenarios = [_scenario(item) for item in SCENARIO_IDS[:2]]
    model = ScriptedAgent(scenarios, error_tasks={scenarios[0].id})
    result, _, _ = _run(tmp_path, scenarios, model)

    assert [trial.status for trial in result.trials] == [
        TrialStatus.AGENT_EXECUTION_ERROR,
        TrialStatus.COMPLETED,
    ]
    assert result.trials[0].run_id is None
    assert result.trials[0].error_type == "ScriptedAgentError"
    assert result.summary.completed_trials == 1
    assert result.summary.agent_execution_error_trials == 1
    assert model.calls.count(scenarios[1].id) > 0


def test_provider_runtime_and_groundedness_provider_failures_are_distinct(tmp_path):
    scenarios = [_scenario(item) for item in SCENARIO_IDS[:2]]
    model = ScriptedAgent(scenarios, provider_error_tasks={scenarios[0].id})
    judge = ScriptedJudge(error=RuntimeError("simulated judge transport error"))
    result, runs, benchmarks = _run(tmp_path, scenarios, model, judge=judge)

    assert result.trials[0].status is TrialStatus.PROVIDER_RUNTIME_ERROR
    assert result.trials[1].status is TrialStatus.COMPLETED
    assert result.summary.provider_runtime_error_trials == 1
    report = BenchmarkReportService(benchmarks, runs).get(result.benchmark_run_id)
    groundedness = next(
        item for item in report.trials[1].run_report.criteria
        if item.criterion_id == "groundedness"
    )
    assert groundedness.status is GraderStatus.INCONCLUSIVE
    assert groundedness.judge_error == "RuntimeError"
    decision = next(
        item for item in report.trials[1].run_report.criteria
        if item.evaluator_name == "decision"
    )
    assert decision.status is GraderStatus.PASS


def test_infrastructure_setup_error_is_not_reported_as_agent_truth(tmp_path):
    class BrokenExecutor:
        def create_runner(self, environment):
            raise RuntimeError("simulated setup failure")

    scenario = _scenario("healthy-canary")
    benchmark = _benchmark([scenario])
    run_store = RunArtifactStore(tmp_path / "runs")
    benchmark_store = BenchmarkArtifactStore(tmp_path / "benchmarks")
    result = BenchmarkRunner(run_store, benchmark_store).run(
        AgentDefinition(
            identity=AgentIdentity(
                agent_id="setup-failure",
                version="1",
                implementation_sha256="c" * 64,
            ),
            executor=BrokenExecutor(),
        ),
        benchmark,
    )

    trial = result.trials[0]
    assert trial.status is TrialStatus.INFRASTRUCTURE_ERROR
    assert trial.error_type == "RuntimeError"
    assert trial.run_id is None
    assert result.summary.completed_trials == 0
    assert result.summary.infrastructure_error_trials == 1
    assert result.summary.criterion_counts == ()


def test_repeated_trials_get_distinct_trial_and_run_artifacts(tmp_path):
    scenario = _scenario(SCENARIO_IDS[0])
    result, runs, _ = _run(
        tmp_path, [scenario], ScriptedAgent([scenario]), trials_per_task=3
    )

    trial_ids = [trial.trial_id for trial in result.trials]
    run_ids = [trial.run_id for trial in result.trials]
    assert len(set(trial_ids)) == len(set(run_ids)) == 3
    assert result.summary.trials_attempted == result.summary.completed_trials == 3
    assert all(runs.load(run_id).trace.scenario_id == scenario.id for run_id in run_ids)


def test_benchmark_artifact_reload_resolves_every_run_reference(tmp_path):
    scenarios = [_scenario(item) for item in SCENARIO_IDS[:2]]
    result, runs, benchmarks = _run(tmp_path, scenarios, ScriptedAgent(scenarios))

    restored = benchmarks.load(result.benchmark_run_id)
    report = BenchmarkReportService(benchmarks, runs).get(result.benchmark_run_id)
    assert restored == result
    assert len(report.trials) == len(scenarios)
    for item in report.trials:
        run = runs.load(item.trial.run_id)
        assert item.run_report.run_id == run.trace.run_id
        assert item.run_report.scenario_id == item.trial.scenario_id


def test_wrong_objective_action_stays_failed_despite_grounded_explanation(tmp_path):
    scenario = _scenario("healthy-canary")
    result, runs, benchmarks = _run(
        tmp_path,
        [scenario],
        ScriptedAgent([scenario], wrong_tasks={scenario.id}),
        judge=ScriptedJudge([GraderStatus.PASS]),
    )

    report = BenchmarkReportService(benchmarks, runs).get(result.benchmark_run_id)
    criteria = report.trials[0].run_report.criteria
    assert (
        next(item for item in criteria if item.evaluator_name == "decision").status
        is GraderStatus.FAIL
    )
    assert (
        next(item for item in criteria if item.criterion_id == "groundedness").status
        is GraderStatus.PASS
    )
    assert not hasattr(report, "overall_status")


def test_benchmark_digest_is_stable_and_semantic_changes_change_it():
    first = _benchmark([_scenario("healthy-canary"), _scenario("degraded-canary")])
    same = _benchmark([_scenario("healthy-canary"), _scenario("degraded-canary")])
    reordered = _benchmark([_scenario("degraded-canary"), _scenario("healthy-canary")])
    changed = first.model_copy(update={"name": "Changed name"})

    assert first.digest == same.digest
    assert first.digest != reordered.digest
    assert first.digest != changed.digest


def test_benchmark_artifact_corruption_and_unsupported_version_fail_safely(tmp_path):
    scenario = _scenario("healthy-canary")
    result, _, benchmarks = _run(tmp_path, [scenario], ScriptedAgent([scenario]))
    path = tmp_path / "benchmarks" / benchmarks._filename(result.benchmark_run_id)
    path.write_text("{malformed", encoding="utf-8")
    with pytest.raises(BenchmarkArtifactCorruptError):
        benchmarks.load(result.benchmark_run_id)

    path.write_text('{"schema_version": 99}', encoding="utf-8")
    with pytest.raises(UnsupportedBenchmarkArtifactVersionError):
        benchmarks.load(result.benchmark_run_id)

    path.write_text('{"schema_version": 1, "result": {}}', encoding="utf-8")
    with pytest.raises(BenchmarkArtifactCorruptError):
        benchmarks.load(result.benchmark_run_id)


def test_benchmark_discovery_ignores_unrelated_files_and_is_sorted(tmp_path):
    scenario = _scenario("healthy-canary")
    one, _, store = _run(tmp_path, [scenario], ScriptedAgent([scenario]))
    two = BenchmarkRunner(RunArtifactStore(tmp_path / "runs"), store).run(
        _agent(ScriptedAgent([scenario])), _benchmark([scenario])
    )
    store.save(two)
    (tmp_path / "benchmarks" / "notes.txt").write_text("unrelated", encoding="utf-8")

    assert [item.benchmark_run_id for item in store.list()] == sorted(
        [one.benchmark_run_id, two.benchmark_run_id]
    )


def test_benchmark_artifact_path_traversal_and_symlinks_are_rejected(tmp_path):
    scenario = _scenario("healthy-canary")
    result, _, store = _run(tmp_path, [scenario], ScriptedAgent([scenario]))
    with pytest.raises(FileNotFoundError):
        store.load("../../outside")

    filename = store._filename("brun-symlink")
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (tmp_path / "benchmarks" / filename).symlink_to(outside)
    with pytest.raises(BenchmarkArtifactCorruptError):
        store.load("brun-symlink")
    assert outside.read_text(encoding="utf-8") == "{}"


def test_atomic_benchmark_publication_failure_leaves_no_final_or_temp_file(
    tmp_path, monkeypatch
):
    scenario = _scenario("healthy-canary")
    result, _, _ = _run(tmp_path, [scenario], ScriptedAgent([scenario]))
    store = BenchmarkArtifactStore(tmp_path / "failed-publication")

    def fail_link(source, destination):
        raise OSError("simulated publication failure")

    monkeypatch.setattr("rollouteval.benchmark.os.link", fail_link)
    with pytest.raises(BenchmarkArtifactWriteError):
        store.save(result)
    assert not (store.root / store._filename(result.benchmark_run_id)).exists()
    assert list(store.root.iterdir()) == []


def test_recruiter_api_can_inspect_trial_and_underlying_run(tmp_path):
    scenario = _scenario("healthy-canary")
    judge = ScriptedJudge([GraderStatus.PASS])
    result, runs, benchmarks = _run(
        tmp_path, [scenario], ScriptedAgent([scenario]), judge=judge
    )
    service = BenchmarkReportService(benchmarks, runs)
    report = service.get(result.benchmark_run_id)
    trial = report.trials[0]
    underlying_run = runs.load(trial.trial.run_id)
    grounded = next(
        item for item in trial.run_report.criteria if item.criterion_id == "groundedness"
    )

    assert report.agent == result.agent
    assert report.benchmark_digest == result.benchmark.digest
    assert trial.run_report.task == scenario.task
    assert grounded.provider == "test-provider"
    assert grounded.model == "test-judge"
    assert grounded.prompt_sha256 == "b" * 64
    assert grounded.evaluator_version == "1"
    assert grounded.criterion_id == "groundedness"
    assert grounded.status is GraderStatus.PASS
    assert grounded.rationale == "Scripted groundedness result."
    assert grounded.evidence_references
    assert underlying_run.trace.run_id == trial.trial.run_id
    assert underlying_run.task.task == scenario.task
    assert trial.trial.agent == report.agent


def test_benchmark_store_idempotency_conflict_and_credential_rejection(tmp_path):
    scenario = _scenario("healthy-canary")
    result, _, store = _run(tmp_path, [scenario], ScriptedAgent([scenario]))

    path = store.save(result)
    assert store.save(result) == path
    changed = result.model_copy(update={"created_at": datetime.now(UTC)})
    with pytest.raises(BenchmarkArtifactConflictError):
        store.save(changed)

    unsafe_agent = AgentIdentity(
        agent_id="OPENAI_API_KEY=sk-1234567890abcdef",
        version=result.agent.version,
        implementation_sha256=result.agent.implementation_sha256,
    )
    unsafe = result.model_copy(
        update={
            "agent": unsafe_agent,
            "trials": tuple(
                trial.model_copy(update={"agent": unsafe_agent}) for trial in result.trials
            ),
        }
    )
    with pytest.raises(BenchmarkArtifactCorruptError, match="credential"):
        store.save(unsafe)
