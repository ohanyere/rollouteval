"""Versioned file persistence and read-only reporting for completed runs."""

import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rollouteval.evaluation.engine import DeterministicEvaluation
from rollouteval.evaluation.results import GraderResult, GraderStatus
from rollouteval.orchestration import ObservableRun

__all__ = [
    "InvalidRunIdentityError",
    "RunArtifactConflictError",
    "RunArtifactCorruptError",
    "RunArtifactEnvelope",
    "RunArtifactError",
    "RunArtifactNotFoundError",
    "RunArtifactSecurityError",
    "RunArtifactStore",
    "RunArtifactWriteError",
    "RunCriterionReport",
    "RunReport",
    "RunReportService",
    "UnsupportedRunArtifactVersionError",
]

_ARTIFACT_SCHEMA_VERSION = 1
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_ARTIFACT_NAME_PATTERN = re.compile(r"run-[0-9a-f]{64}\.json\Z")
_SECRET_PATTERNS = (
    re.compile(
        r"(?i)\b(?:OPENAI_API_KEY|GROQ_API_KEY)\b\s*[:=]\s*['\"]?[A-Za-z0-9_-]{8,}"
    ),
    re.compile(
        r"(?i)\bauthorization\b.{0,12}[:=].{0,48}\b(?:bearer|basic)\s+\S+"
    ),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\b(?:sk|gsk)[_-][A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
)


class RunArtifactError(Exception):
    """Base error for invalid or inaccessible run artifacts."""


class InvalidRunIdentityError(RunArtifactError, ValueError):
    """Raised when a run identity cannot safely address an artifact."""


class RunArtifactNotFoundError(RunArtifactError, FileNotFoundError):
    """Raised when no artifact exists for a run identity."""


class RunArtifactConflictError(RunArtifactError):
    """Raised when an identity already stores a different completed run."""


class RunArtifactCorruptError(RunArtifactError):
    """Raised when an artifact is malformed or internally inconsistent."""


class UnsupportedRunArtifactVersionError(RunArtifactCorruptError):
    """Raised when an artifact uses an unsupported schema version."""


class RunArtifactSecurityError(RunArtifactError):
    """Raised when serialized content appears to contain credential material."""


class RunArtifactWriteError(RunArtifactError):
    """Raised when a run artifact cannot be safely written."""


class RunArtifactEnvelope(BaseModel):
    """Versioned metadata wrapping the canonical completed-run model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_schema_version: Literal[1]
    rollouteval_version: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    task: str = Field(min_length=1)
    created_at: datetime
    observable_run: ObservableRun

    @model_validator(mode="before")
    @classmethod
    def require_complete_payload(cls, value):
        if not isinstance(value, dict):
            return value
        payload = value.get("observable_run")
        if isinstance(payload, ObservableRun):
            return value
        if not isinstance(payload, dict):
            raise ValueError("observable_run must be a complete object")
        missing_run_fields = set(ObservableRun.model_fields) - payload.keys()
        if missing_run_fields:
            raise ValueError("observable_run is missing required fields")
        evaluation = payload.get("evaluation")
        if not isinstance(evaluation, dict):
            raise ValueError("evaluation must be a complete object")
        missing_evaluation_fields = (
            set(DeterministicEvaluation.model_fields) - evaluation.keys()
        )
        if missing_evaluation_fields:
            raise ValueError("evaluation is missing required fields")
        for result in evaluation.values():
            if result is None:
                continue
            if not isinstance(result, dict):
                raise ValueError("criterion results must be objects")
            expected_result_fields = set(GraderResult.model_fields)
            if expected_result_fields - result.keys():
                raise ValueError("criterion result is missing required fields")
        return value

    @model_validator(mode="after")
    def validate_run_identity(self) -> "RunArtifactEnvelope":
        run = self.observable_run
        if self.run_id != run.trace.run_id:
            raise ValueError("run_id does not match ObservableRun trace")
        if self.scenario_id != run.trace.scenario_id:
            raise ValueError("scenario_id does not match ObservableRun trace")
        if self.task != run.task.task:
            raise ValueError("task does not match ObservableRun task")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return self


class RunCriterionReport(BaseModel):
    """One faithfully projected criterion result, without aggregation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluator_name: str
    evaluator_version: str
    criterion_id: str | None
    status: GraderStatus
    rationale: str
    evidence_references: tuple[str, ...]
    provider: str | None
    model: str | None
    prompt_sha256: str | None
    judge_error: str | None


class RunReport(BaseModel):
    """Read-only report for one persisted run and its independent criteria."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    scenario_id: str
    task: str
    criteria: tuple[RunCriterionReport, ...]


def _package_version() -> str:
    try:
        return version("rollouteval")
    except PackageNotFoundError:
        return "unknown"


def _artifact_filename(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise InvalidRunIdentityError("run_id contains unsupported path characters")
    digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    return f"run-{digest}.json"


def _reject_secrets(serialized: str) -> None:
    if any(pattern.search(serialized) for pattern in _SECRET_PATTERNS):
        raise RunArtifactSecurityError("artifact appears to contain credential material")


class RunArtifactStore:
    """Explicit-root, atomic file storage for completed ObservableRun objects."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()

    def save(self, run: ObservableRun) -> Path:
        """Atomically save a run; identical repeats are idempotent."""
        validated_run = ObservableRun.model_validate(run.model_dump(mode="python"))
        run_id = validated_run.trace.run_id
        filename = _artifact_filename(run_id)
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / filename
        if destination.exists() or destination.is_symlink():
            existing = self._load_path(destination, expected_filename=filename)
            if existing.observable_run == validated_run:
                return destination
            raise RunArtifactConflictError("run_id already stores a different run")

        envelope = RunArtifactEnvelope(
            artifact_schema_version=_ARTIFACT_SCHEMA_VERSION,
            rollouteval_version=_package_version(),
            run_id=run_id,
            scenario_id=validated_run.trace.scenario_id,
            task=validated_run.task.task,
            created_at=datetime.now(UTC),
            observable_run=validated_run,
        )
        serialized = json.dumps(
            envelope.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        _reject_secrets(serialized)
        self._atomic_create(destination, serialized.encode("utf-8"), validated_run)
        return destination

    def load(self, run_id: str) -> ObservableRun:
        """Load and validate a completed run by its stable trace identity."""
        filename = _artifact_filename(run_id)
        path = self.root / filename
        if not path.exists() and not path.is_symlink():
            raise RunArtifactNotFoundError("no run artifact exists for run_id")
        return self._load_path(path, expected_filename=filename).observable_run

    def list(self) -> tuple[ObservableRun, ...]:
        """List recognized artifacts in deterministic run-identity order."""
        if not self.root.exists():
            return ()
        runs = []
        for path in sorted(self.root.iterdir(), key=lambda item: item.name):
            if not _ARTIFACT_NAME_PATTERN.fullmatch(path.name):
                continue
            runs.append(self._load_path(path, expected_filename=path.name).observable_run)
        identities = [run.trace.run_id for run in runs]
        if len(identities) != len(set(identities)):
            raise RunArtifactCorruptError("duplicate run identities were discovered")
        return tuple(sorted(runs, key=lambda run: run.trace.run_id))

    def _load_path(self, path: Path, *, expected_filename: str) -> RunArtifactEnvelope:
        if path.is_symlink() or not path.is_file():
            raise RunArtifactCorruptError("artifact path is not a regular file")
        try:
            serialized = path.read_text(encoding="utf-8")
            _reject_secrets(serialized)
            value = json.loads(serialized)
        except RunArtifactSecurityError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RunArtifactCorruptError("artifact is not valid UTF-8 JSON") from exc
        if not isinstance(value, dict):
            raise RunArtifactCorruptError("artifact envelope must be a JSON object")
        schema_version = value.get("artifact_schema_version")
        if type(schema_version) is not int or schema_version != _ARTIFACT_SCHEMA_VERSION:
            raise UnsupportedRunArtifactVersionError(
                f"unsupported artifact schema version: {schema_version!r}"
            )
        try:
            envelope = RunArtifactEnvelope.model_validate(value)
            actual_filename = _artifact_filename(envelope.run_id)
        except (ValidationError, ValueError, TypeError) as exc:
            raise RunArtifactCorruptError("artifact structure or identity is invalid") from exc
        if actual_filename != expected_filename:
            raise RunArtifactCorruptError("artifact filename does not match run identity")
        return envelope

    def _atomic_create(
        self,
        destination: Path,
        content: bytes,
        run: ObservableRun,
    ) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".run-artifact-", suffix=".tmp", dir=self.root
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as temporary_file:
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            try:
                os.link(temporary_path, destination)
            except FileExistsError:
                existing = self._load_path(
                    destination,
                    expected_filename=destination.name,
                )
                if existing.observable_run != run:
                    raise RunArtifactConflictError(
                        "run_id already stores a different run"
                    )
                return
            directory_descriptor = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except RunArtifactError:
            raise
        except OSError as exc:
            raise RunArtifactWriteError("artifact could not be written atomically") from exc
        finally:
            temporary_path.unlink(missing_ok=True)


class RunReportService:
    """Reporting read path backed by the normal run-artifact store."""

    def __init__(self, store: RunArtifactStore) -> None:
        self._store = store

    def get(self, run_id: str) -> RunReport:
        run = self._store.load(run_id)
        criteria = tuple(
            RunCriterionReport(
                evaluator_name=result.grader,
                evaluator_version=result.evaluator_version,
                criterion_id=result.criterion_id,
                status=result.status,
                rationale=result.reason,
                evidence_references=result.evidence_references,
                provider=result.judge_provider,
                model=result.judge_model,
                prompt_sha256=result.prompt_sha256,
                judge_error=result.judge_error,
            )
            for result in run.evaluation.results()
        )
        return RunReport(
            run_id=run.trace.run_id,
            scenario_id=run.trace.scenario_id,
            task=run.task.task,
            criteria=criteria,
        )

    def list(self) -> tuple[RunReport, ...]:
        return tuple(self.get(run.trace.run_id) for run in self._store.list())
