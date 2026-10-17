"""Provider-neutral benchmark execution, persistence, and reporting."""

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rollouteval.agent.runner import (
    AgentModel,
    AgentRunExecutor,
    AgentRunner,
    AgentStepLimitExceeded,
)
from rollouteval.domain.scenario import Scenario
from rollouteval.environment import RolloutEnvironment
from rollouteval.evaluation.groundedness import GroundednessJudge
from rollouteval.evaluation.results import GraderStatus
from rollouteval.observability.trace import TraceRecorder
from rollouteval.orchestration import ObservableRunOrchestrator
from rollouteval.run_artifacts import RunArtifactStore, RunReport, RunReportService
from rollouteval.tools import RolloutMutationTools, RolloutObservationTools

_ARTIFACT_SCHEMA_VERSION = 1
_ARTIFACT_NAME = re.compile(r"benchmark-[0-9a-f]{64}\.json\Z")
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(?:OPENAI_API_KEY|GROQ_API_KEY)\b\s*[:=]\s*['\"]?[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)\bauthorization\b.{0,12}[:=].{0,48}\b(?:bearer|basic)\s+\S+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\b(?:sk|gsk)[_-][A-Za-z0-9_-]{8,}\b"),
)


class AgentIdentity(BaseModel):
    """Stable identity for one immutable agent implementation version."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    agent_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=128)
    implementation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    connection_type: Literal["in_process", "local_subprocess", "remote_http"] = "in_process"
    protocol_version: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_connection_provenance(self) -> "AgentIdentity":
        is_external = self.connection_type in {"local_subprocess", "remote_http"}
        if is_external != (self.protocol_version is not None):
            raise ValueError(
                "external agent identities require a protocol version"
            )
        return self


class BenchmarkSpec(BaseModel):
    """Ordered, immutable benchmark definition over validated scenarios."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    benchmark_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=128)
    tasks: tuple[Scenario, ...] = Field(min_length=1)
    name: str | None = None
    description: str | None = None

    @model_validator(mode="after")
    def unique_task_ids(self) -> "BenchmarkSpec":
        identities = [scenario.id for scenario in self.tasks]
        if len(identities) != len(set(identities)):
            raise ValueError("benchmark task scenario IDs must be unique")
        return self

    @property
    def digest(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class TrialStatus(StrEnum):
    COMPLETED = "COMPLETED"
    AGENT_EXECUTION_ERROR = "AGENT_EXECUTION_ERROR"
    PROVIDER_RUNTIME_ERROR = "PROVIDER_RUNTIME_ERROR"
    INFRASTRUCTURE_ERROR = "INFRASTRUCTURE_ERROR"


class CriterionCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion_id: str | None
    evaluator_name: str
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    inconclusive: int = Field(ge=0)


class BenchmarkSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tasks_attempted: int = Field(ge=0)
    trials_attempted: int = Field(ge=0)
    completed_trials: int = Field(ge=0)
    agent_execution_error_trials: int = Field(ge=0)
    provider_runtime_error_trials: int = Field(ge=0)
    infrastructure_error_trials: int = Field(ge=0)
    criterion_counts: tuple[CriterionCounts, ...]

    @model_validator(mode="after")
    def trial_counts_are_consistent(self) -> "BenchmarkSummary":
        if self.trials_attempted != (
            self.completed_trials
            + self.agent_execution_error_trials
            + self.provider_runtime_error_trials
            + self.infrastructure_error_trials
        ):
            raise ValueError("trial summary counts do not add up")
        keys = [
            (item.criterion_id, item.evaluator_name)
            for item in self.criterion_counts
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("criterion counts must have unique identities")
        if any(
            item.passed + item.failed + item.inconclusive > self.completed_trials
            for item in self.criterion_counts
        ):
            raise ValueError("criterion counts exceed completed trials")
        return self


class TrialRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trial_id: str = Field(min_length=1)
    benchmark_id: str
    benchmark_version: str
    benchmark_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    agent: AgentIdentity
    scenario_id: str
    ordinal: int = Field(ge=1)
    run_id: str | None = None
    status: TrialStatus
    error_type: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_.]{0,127}$")
    started_at: datetime
    completed_at: datetime

    @model_validator(mode="after")
    def validate_status_fields(self) -> "TrialRecord":
        has_run = self.run_id is not None
        has_error = self.error_type is not None
        if (self.status is TrialStatus.COMPLETED) != has_run:
            raise ValueError("only completed trials must reference an ObservableRun")
        if (self.status is TrialStatus.COMPLETED) == has_error:
            raise ValueError("failed trials require an error type; completed trials do not")
        if self.started_at.tzinfo is None or self.completed_at.tzinfo is None:
            raise ValueError("trial timestamps must be timezone-aware")
        return self


class BenchmarkRunResult(BaseModel):
    """Immutable benchmark execution record; run artifacts remain canonical."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_run_id: str = Field(min_length=1)
    benchmark: BenchmarkSpec
    benchmark_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    agent: AgentIdentity
    trials_per_task: int = Field(ge=1)
    trials: tuple[TrialRecord, ...]
    summary: BenchmarkSummary
    created_at: datetime

    @model_validator(mode="after")
    def validate_run(self) -> "BenchmarkRunResult":
        if self.benchmark_digest != self.benchmark.digest:
            raise ValueError("benchmark digest does not match its definition")
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
        expected_trials = len(self.benchmark.tasks) * self.trials_per_task
        if len(self.trials) != expected_trials:
            raise ValueError("benchmark run does not contain every requested trial")
        expected_order = [
            (task.id, ordinal)
            for task in self.benchmark.tasks
            for ordinal in range(1, self.trials_per_task + 1)
        ]
        actual_order = [(trial.scenario_id, trial.ordinal) for trial in self.trials]
        if actual_order != expected_order:
            raise ValueError("trials do not match deterministic benchmark order")
        trial_ids = [trial.trial_id for trial in self.trials]
        run_ids = [trial.run_id for trial in self.trials if trial.run_id is not None]
        if len(trial_ids) != len(set(trial_ids)) or len(run_ids) != len(set(run_ids)):
            raise ValueError("trial and run identities must be unique")
        if any(
            not trial.trial_id.startswith(f"{self.benchmark_run_id}-trial-")
            or (trial.run_id is not None and trial.run_id != f"{trial.trial_id}-run")
            for trial in self.trials
        ):
            raise ValueError("trial and run identities do not match benchmark run")
        if any(
            trial.benchmark_id != self.benchmark.benchmark_id
            or trial.benchmark_version != self.benchmark.version
            or trial.benchmark_digest != self.benchmark_digest
            or trial.agent != self.agent
            for trial in self.trials
        ):
            raise ValueError("trial provenance does not match benchmark run")
        if self.summary.trials_attempted != len(self.trials):
            raise ValueError("summary does not match trial records")
        if self.summary.tasks_attempted != len({trial.scenario_id for trial in self.trials}):
            raise ValueError("summary task count does not match trial records")
        expected = _summary_trial_counts(self.trials)
        actual = (
            self.summary.completed_trials,
            self.summary.agent_execution_error_trials,
            self.summary.provider_runtime_error_trials,
            self.summary.infrastructure_error_trials,
        )
        if actual != expected:
            raise ValueError("summary does not match trial statuses")
        return self


class BenchmarkArtifactError(Exception):
    """Base benchmark artifact error."""


class BenchmarkArtifactCorruptError(BenchmarkArtifactError):
    """Raised when a benchmark artifact is malformed or inconsistent."""


class BenchmarkArtifactConflictError(BenchmarkArtifactError):
    """Raised when a benchmark-run identity is reused for different content."""


class BenchmarkArtifactWriteError(BenchmarkArtifactError):
    """Raised when atomic benchmark artifact publication fails."""


class UnsupportedBenchmarkArtifactVersionError(BenchmarkArtifactCorruptError):
    """Raised for unsupported benchmark artifact schema versions."""


class BenchmarkArtifactEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    result: BenchmarkRunResult


class BenchmarkArtifactStore:
    """Explicit-root, atomic persistence for benchmark-run results."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()

    @staticmethod
    def _filename(benchmark_run_id: str) -> str:
        if not benchmark_run_id or len(benchmark_run_id) > 128:
            raise ValueError("invalid benchmark_run_id")
        digest = hashlib.sha256(benchmark_run_id.encode("utf-8")).hexdigest()
        return f"benchmark-{digest}.json"

    def save(self, result: BenchmarkRunResult) -> Path:
        result = BenchmarkRunResult.model_validate(result.model_dump(mode="python"))
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise BenchmarkArtifactWriteError(
                "benchmark artifact root could not be created"
            ) from exc
        destination = self.root / self._filename(result.benchmark_run_id)
        payload = json.dumps(
            BenchmarkArtifactEnvelope(schema_version=1, result=result).model_dump(
                mode="json"
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        _reject_credentials(payload)
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink():
                raise BenchmarkArtifactCorruptError("artifact path is a symlink")
            existing = self._load(destination, destination.name)
            if existing == result:
                return destination
            raise BenchmarkArtifactConflictError(
                "benchmark_run_id already stores different content"
            )
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".benchmark-artifact-", suffix=".tmp", dir=self.root
            )
        except OSError as exc:
            raise BenchmarkArtifactWriteError(
                "benchmark artifact temporary file could not be created"
            ) from exc
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as artifact:
                artifact.write(payload)
                artifact.flush()
                os.fsync(artifact.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                existing = self._load(destination, destination.name)
                if existing != result:
                    raise BenchmarkArtifactConflictError(
                        "benchmark_run_id already stores different content"
                    )
            directory = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BenchmarkArtifactError:
            raise
        except OSError as exc:
            raise BenchmarkArtifactWriteError(
                "benchmark artifact could not be published atomically"
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def load(self, benchmark_run_id: str) -> BenchmarkRunResult:
        filename = self._filename(benchmark_run_id)
        path = self.root / filename
        if path.is_symlink():
            raise BenchmarkArtifactCorruptError("artifact path is a symlink")
        if not path.is_file():
            raise FileNotFoundError("benchmark run artifact does not exist")
        return self._load(path, filename)

    def list(self) -> tuple[BenchmarkRunResult, ...]:
        if not self.root.exists():
            return ()
        results = []
        for path in sorted(self.root.iterdir(), key=lambda item: item.name):
            if _ARTIFACT_NAME.fullmatch(path.name):
                if path.is_symlink() or not path.is_file():
                    raise BenchmarkArtifactCorruptError("artifact is not a regular file")
                results.append(self._load(path, path.name))
        identities = [result.benchmark_run_id for result in results]
        if len(identities) != len(set(identities)):
            raise BenchmarkArtifactCorruptError("duplicate benchmark run identity")
        return tuple(sorted(results, key=lambda result: result.benchmark_run_id))

    @staticmethod
    def _load(path: Path, expected_filename: str) -> BenchmarkRunResult:
        if path.is_symlink() or not path.is_file():
            raise BenchmarkArtifactCorruptError("artifact is not a regular file")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            _reject_credentials(json.dumps(value))
            if not isinstance(value, dict):
                raise ValueError("artifact must be an object")
            if type(value.get("schema_version")) is not int or value.get("schema_version") != 1:
                raise UnsupportedBenchmarkArtifactVersionError(
                    "unsupported benchmark artifact schema version"
                )
            envelope = BenchmarkArtifactEnvelope.model_validate(value)
        except UnsupportedBenchmarkArtifactVersionError:
            raise
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValidationError,
            ValueError,
        ) as exc:
            raise BenchmarkArtifactCorruptError("benchmark artifact is invalid") from exc
        if BenchmarkArtifactStore._filename(envelope.result.benchmark_run_id) != expected_filename:
            raise BenchmarkArtifactCorruptError("artifact filename identity mismatch")
        return envelope.result


def _reject_credentials(serialized: str) -> None:
    if any(pattern.search(serialized) for pattern in _SECRET_PATTERNS):
        raise BenchmarkArtifactCorruptError("artifact contains credential-like content")


class AgentExecutor(Protocol):
    """Provider-neutral factory for an isolated runner bound to a task runtime."""

    def create_runner(self, environment: RolloutEnvironment) -> AgentRunExecutor:
        ...


@dataclass(frozen=True)
class ModelAgentExecutor:
    """Adapt the existing provider-neutral AgentModel interface to each trial."""

    model: AgentModel
    max_steps: int = 8

    def create_runner(self, environment: RolloutEnvironment) -> AgentRunner:
        observations = RolloutObservationTools(environment)
        mutations = RolloutMutationTools(environment)
        return AgentRunner(
            _ProviderBoundaryModel(self.model),
            observations,
            mutations,
            max_steps=self.max_steps,
        )


class ProviderRuntimeError(RuntimeError):
    """Provider-neutral marker for runtime failures raised by an agent adapter."""

    def __init__(self, error_type: str) -> None:
        safe_type = (
            error_type
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,127}", error_type)
            else "ProviderRuntimeError"
        )
        super().__init__(safe_type)
        self.error_type = safe_type


class AgentExecutionError(RuntimeError):
    """Provider-neutral marker for agent-side execution failures."""

    def __init__(self, error_type: str) -> None:
        safe_type = (
            error_type
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,127}", error_type)
            else "AgentExecutionError"
        )
        super().__init__(safe_type)
        self.error_type = safe_type


class _ProviderBoundaryModel:
    def __init__(self, model: AgentModel) -> None:
        self._model = model

    def respond(self, task, interactions):
        try:
            return self._model.respond(task, interactions)
        except (ProviderRuntimeError, AgentExecutionError):
            raise
        except Exception as exc:
            raise ProviderRuntimeError(type(exc).__name__) from exc


@dataclass(frozen=True)
class AgentDefinition:
    """Stable agent-version identity paired with a provider-neutral executor."""

    identity: AgentIdentity
    executor: AgentExecutor


class _ClassifiedRunner:
    def __init__(self, runner: AgentRunExecutor) -> None:
        self._runner = runner

    def run(self, task, *, trace_recorder=None):
        try:
            return self._runner.run(task, trace_recorder=trace_recorder)
        except (AgentExecutionError, ProviderRuntimeError):
            raise
        except AgentStepLimitExceeded as exc:
            raise AgentExecutionError(type(exc).__name__) from exc
        except Exception as exc:
            raise AgentExecutionError(type(exc).__name__) from exc


class BenchmarkRunner:
    """Execute ordered trials, preserving each run and continuing after errors."""

    def __init__(
        self,
        run_store: RunArtifactStore,
        benchmark_store: BenchmarkArtifactStore,
        *,
        orchestrator: ObservableRunOrchestrator | None = None,
    ) -> None:
        self._run_store = run_store
        self._benchmark_store = benchmark_store
        self._orchestrator = orchestrator or ObservableRunOrchestrator()

    def run(
        self,
        agent: AgentDefinition,
        benchmark: BenchmarkSpec,
        *,
        trials_per_task: int = 1,
        groundedness_judge: GroundednessJudge | None = None,
    ) -> BenchmarkRunResult:
        if trials_per_task < 1:
            raise ValueError("trials_per_task must be at least one")
        benchmark_run_id = f"brun-{uuid4().hex}"
        records: list[TrialRecord] = []
        completed_runs = []
        ordinal_global = 0
        for scenario in benchmark.tasks:
            for ordinal in range(1, trials_per_task + 1):
                ordinal_global += 1
                trial_id = f"{benchmark_run_id}-trial-{ordinal_global}"
                run_id = f"{trial_id}-run"
                started_at = datetime.now(UTC)
                try:
                    environment = RolloutEnvironment(scenario)
                    recorder = TraceRecorder(run_id=run_id, scenario_id=scenario.id)
                    runner = _ClassifiedRunner(
                        agent.executor.create_runner(environment)
                    )
                    observable = self._orchestrator.execute(
                        scenario=scenario,
                        runner=runner,
                        environment=environment,
                        trace_recorder=recorder,
                        groundedness_judge=groundedness_judge,
                    )
                    self._run_store.save(observable)
                except Exception as exc:
                    if isinstance(exc, ProviderRuntimeError):
                        status = TrialStatus.PROVIDER_RUNTIME_ERROR
                    elif isinstance(exc, AgentExecutionError):
                        status = TrialStatus.AGENT_EXECUTION_ERROR
                    else:
                        status = TrialStatus.INFRASTRUCTURE_ERROR
                    records.append(
                        TrialRecord(
                            trial_id=trial_id,
                            benchmark_id=benchmark.benchmark_id,
                            benchmark_version=benchmark.version,
                            benchmark_digest=benchmark.digest,
                            agent=agent.identity,
                            scenario_id=scenario.id,
                            ordinal=ordinal,
                            run_id=None,
                            status=status,
                            error_type=getattr(exc, "error_type", type(exc).__name__),
                            started_at=started_at,
                            completed_at=datetime.now(UTC),
                        )
                    )
                    continue
                completed_runs.append(observable)
                records.append(
                    TrialRecord(
                        trial_id=trial_id,
                        benchmark_id=benchmark.benchmark_id,
                        benchmark_version=benchmark.version,
                        benchmark_digest=benchmark.digest,
                        agent=agent.identity,
                        scenario_id=scenario.id,
                        ordinal=ordinal,
                        run_id=run_id,
                        status=TrialStatus.COMPLETED,
                        started_at=started_at,
                        completed_at=datetime.now(UTC),
                    )
                )
        summary = _summarize(records, completed_runs)
        result = BenchmarkRunResult(
            benchmark_run_id=benchmark_run_id,
            benchmark=benchmark,
            benchmark_digest=benchmark.digest,
            agent=agent.identity,
            trials_per_task=trials_per_task,
            trials=tuple(records),
            summary=summary,
            created_at=datetime.now(UTC),
        )
        self._benchmark_store.save(result)
        return result


def _summarize(
    trials: list[TrialRecord],
    runs: list,
) -> BenchmarkSummary:
    counts: dict[tuple[str | None, str], list[int]] = {}
    for run in runs:
        for criterion in run.evaluation.results():
            key = (criterion.criterion_id, criterion.grader)
            tally = counts.setdefault(key, [0, 0, 0])
            index = {
                GraderStatus.PASS: 0,
                GraderStatus.FAIL: 1,
                GraderStatus.INCONCLUSIVE: 2,
            }[criterion.status]
            tally[index] += 1
    task_ids = {trial.scenario_id for trial in trials}
    return BenchmarkSummary(
        tasks_attempted=len(task_ids),
        trials_attempted=len(trials),
        completed_trials=sum(trial.status is TrialStatus.COMPLETED for trial in trials),
        agent_execution_error_trials=sum(
            trial.status is TrialStatus.AGENT_EXECUTION_ERROR for trial in trials
        ),
        provider_runtime_error_trials=sum(
            trial.status is TrialStatus.PROVIDER_RUNTIME_ERROR for trial in trials
        ),
        infrastructure_error_trials=sum(
            trial.status is TrialStatus.INFRASTRUCTURE_ERROR for trial in trials
        ),
        criterion_counts=tuple(
            CriterionCounts(
                criterion_id=criterion_id,
                evaluator_name=evaluator,
                passed=tally[0],
                failed=tally[1],
                inconclusive=tally[2],
            )
            for (criterion_id, evaluator), tally in sorted(
                counts.items(), key=lambda item: (item[0][0] or "", item[0][1])
            )
        ),
    )


def _summary_trial_counts(trials: tuple[TrialRecord, ...]) -> tuple[int, int, int, int]:
    return (
        sum(trial.status is TrialStatus.COMPLETED for trial in trials),
        sum(trial.status is TrialStatus.AGENT_EXECUTION_ERROR for trial in trials),
        sum(trial.status is TrialStatus.PROVIDER_RUNTIME_ERROR for trial in trials),
        sum(trial.status is TrialStatus.INFRASTRUCTURE_ERROR for trial in trials),
    )


class TrialReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trial: TrialRecord
    run_report: RunReport | None


class BenchmarkReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_run_id: str
    benchmark_id: str
    benchmark_version: str
    benchmark_digest: str
    agent: AgentIdentity
    tasks_attempted: int
    trials_attempted: int
    completed_trials: int
    agent_execution_error_trials: int
    provider_runtime_error_trials: int
    infrastructure_error_trials: int
    criterion_counts: tuple[CriterionCounts, ...]
    trials: tuple[TrialReport, ...]


class BenchmarkReportService:
    """Read benchmark summaries and resolve completed trial run reports."""

    def __init__(
        self,
        benchmark_store: BenchmarkArtifactStore,
        run_store: RunArtifactStore,
    ) -> None:
        self._benchmark_store = benchmark_store
        self._run_reports = RunReportService(run_store)

    def get(self, benchmark_run_id: str) -> BenchmarkReport:
        result = self._benchmark_store.load(benchmark_run_id)
        reports = tuple(
            TrialReport(
                trial=trial,
                run_report=(
                    self._run_reports.get(trial.run_id)
                    if trial.run_id is not None
                    else None
                ),
            )
            for trial in result.trials
        )
        for item in reports:
            if item.run_report is not None and (
                item.run_report.run_id != item.trial.run_id
                or item.run_report.scenario_id != item.trial.scenario_id
            ):
                raise BenchmarkArtifactCorruptError(
                    "trial reference does not match the persisted run artifact"
                )
        return BenchmarkReport(
            benchmark_run_id=result.benchmark_run_id,
            benchmark_id=result.benchmark.benchmark_id,
            benchmark_version=result.benchmark.version,
            benchmark_digest=result.benchmark_digest,
            agent=result.agent,
            tasks_attempted=result.summary.tasks_attempted,
            trials_attempted=result.summary.trials_attempted,
            completed_trials=result.summary.completed_trials,
            agent_execution_error_trials=result.summary.agent_execution_error_trials,
            provider_runtime_error_trials=result.summary.provider_runtime_error_trials,
            infrastructure_error_trials=result.summary.infrastructure_error_trials,
            criterion_counts=result.summary.criterion_counts,
            trials=reports,
        )

    def list(self) -> tuple[BenchmarkReport, ...]:
        return tuple(
            self.get(result.benchmark_run_id)
            for result in self._benchmark_store.list()
        )
