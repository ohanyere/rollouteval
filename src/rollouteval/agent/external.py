"""Versioned local subprocess boundary for external RolloutEval agents."""

import json
import os
import signal
import subprocess
import threading
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rollouteval.agent.contracts import (
    AgentModelResponse,
    AgentTask,
    AgentToolInteraction,
)
from rollouteval.agent.runner import AgentModel, AgentRunExecutor, AgentRunner
from rollouteval.benchmark import (
    AgentDefinition,
    AgentExecutionError,
    AgentIdentity,
)
from rollouteval.environment import RolloutEnvironment
from rollouteval.observability.trace import TraceRecorder
from rollouteval.tools import RolloutMutationTools, RolloutObservationTools

EXTERNAL_AGENT_PROTOCOL_VERSION = 1
_MAX_COMMAND_ARGS = 32
_MAX_ARGUMENT_LENGTH = 4096
_MAX_ENVIRONMENT_VARIABLES = 32
_MAX_ENVIRONMENT_VALUE_LENGTH = 8192


class ExternalEnvironmentVariable(BaseModel):
    """One explicitly allowlisted child-process environment variable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
    value: str = Field(max_length=_MAX_ENVIRONMENT_VALUE_LENGTH)


class ExternalAgentSpec(BaseModel):
    """Immutable local command and bounded process/protocol configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=128)
    implementation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    command: tuple[str, ...] = Field(min_length=1, max_length=_MAX_COMMAND_ARGS)
    working_directory: Path | None = None
    environment: tuple[ExternalEnvironmentVariable, ...] = Field(
        default=(), max_length=_MAX_ENVIRONMENT_VARIABLES
    )
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    max_request_bytes: int = Field(default=1_048_576, ge=1, le=8_388_608)
    max_stdout_bytes: int = Field(default=1_048_576, ge=1, le=8_388_608)
    max_stderr_bytes: int = Field(default=65_536, ge=1, le=1_048_576)
    max_steps: int = Field(default=8, ge=1, le=64)
    protocol_version: Literal[1] = EXTERNAL_AGENT_PROTOCOL_VERSION

    @model_validator(mode="after")
    def validate_command_and_environment(self) -> "ExternalAgentSpec":
        if any(
            not argument
            or len(argument) > _MAX_ARGUMENT_LENGTH
            or "\x00" in argument
            for argument in self.command
        ):
            raise ValueError("command arguments must be non-empty, bounded, and NUL-free")
        names = [variable.name for variable in self.environment]
        if len(names) != len(set(names)):
            raise ValueError("environment variable names must be unique")
        return self

    @property
    def child_environment(self) -> dict[str, str]:
        """Return only explicitly configured values; never inherit the parent env."""
        return {variable.name: variable.value for variable in self.environment}


class ExternalAgentRequest(BaseModel):
    """One task/model-turn request sent to the external process over stdin."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: Literal[1]
    request_id: str
    scenario_id: str
    run_id: str | None
    task: str
    interactions: tuple[AgentToolInteraction, ...]


class ExternalAgentResponse(BaseModel):
    """One validated AgentModel response returned over stdout."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: Literal[1]
    response: AgentModelResponse


class ExternalAgentError(AgentExecutionError):
    """Base safe failure class for local external-agent execution."""

    def __init__(self, error_type: str | None = None) -> None:
        super().__init__(error_type or type(self).__name__)


class ExternalAgentExecutableNotFoundError(ExternalAgentError):
    """The configured executable could not be found."""


class ExternalAgentLaunchError(ExternalAgentError):
    """The configured process could not be launched."""


class ExternalAgentTimeoutError(ExternalAgentError):
    """The external process exceeded its execution timeout."""


class ExternalAgentNonZeroExitError(ExternalAgentError):
    """The external process exited unsuccessfully."""


class ExternalAgentMalformedJSONError(ExternalAgentError):
    """The process did not write one valid UTF-8 JSON response."""


class UnsupportedExternalAgentProtocolError(ExternalAgentError):
    """The external agent returned an unsupported protocol version."""


class InvalidExternalAgentResponseError(ExternalAgentError):
    """The JSON response did not match the versioned response schema."""


class ExternalAgentOutputLimitError(ExternalAgentError):
    """The external process exceeded a configured stdout or stderr bound."""


class ExternalAgentStdoutLimitError(ExternalAgentOutputLimitError):
    """The external process exceeded its stdout bound."""


class ExternalAgentStderrLimitError(ExternalAgentOutputLimitError):
    """The external process exceeded its stderr bound."""


class ExternalAgentRequestLimitError(ExternalAgentError):
    """The serialized request exceeded the configured input bound."""


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        elif process.poll() is None:
            process.kill()
    except ProcessLookupError:
        pass
    except OSError:
        try:
            process.kill()
        except OSError:
            pass


def _run_command(spec: ExternalAgentSpec, request_bytes: bytes) -> bytes:
    if len(request_bytes) > spec.max_request_bytes:
        raise ExternalAgentRequestLimitError()
    try:
        process = subprocess.Popen(
            spec.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=spec.working_directory,
            env=spec.child_environment,
            shell=False,
            close_fds=True,
            start_new_session=os.name == "posix",
        )
    except FileNotFoundError as exc:
        raise ExternalAgentExecutableNotFoundError() from exc
    except OSError as exc:
        raise ExternalAgentLaunchError() from exc

    stdout = bytearray()
    stderr = bytearray()
    failures: SimpleQueue[ExternalAgentOutputLimitError] = SimpleQueue()
    deadline = time.monotonic() + spec.timeout_seconds

    def read_bounded(stream, target: bytearray, limit: int, error_type) -> None:
        try:
            while True:
                remaining = limit + 1 - len(target)
                chunk = stream.read1(min(65_536, remaining))
                if not chunk:
                    return
                target.extend(chunk)
                if len(target) > limit:
                    failures.put(error_type())
                    _terminate_process(process)
                    return
        except OSError:
            return

    def write_request() -> None:
        try:
            process.stdin.write(request_bytes)
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass

    stdout_thread = threading.Thread(
        target=read_bounded,
        args=(process.stdout, stdout, spec.max_stdout_bytes, ExternalAgentStdoutLimitError),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=read_bounded,
        args=(process.stderr, stderr, spec.max_stderr_bytes, ExternalAgentStderrLimitError),
        daemon=True,
    )
    input_thread = threading.Thread(target=write_request, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    input_thread.start()

    try:
        process.wait(timeout=spec.timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        _terminate_process(process)
        process.wait()
        for thread in (input_thread, stdout_thread, stderr_thread):
            thread.join(1)
        raise ExternalAgentTimeoutError() from exc

    for thread in (input_thread, stdout_thread, stderr_thread):
        thread.join(max(0, deadline - time.monotonic()))
    if any(thread.is_alive() for thread in (input_thread, stdout_thread, stderr_thread)):
        _terminate_process(process)
        process.wait()
        raise ExternalAgentTimeoutError()
    if not failures.empty():
        raise failures.get()
    if process.returncode != 0:
        raise ExternalAgentNonZeroExitError()
    return bytes(stdout)


class _SubprocessAgentModel(AgentModel):
    def __init__(self, spec: ExternalAgentSpec, scenario_id: str) -> None:
        self._spec = spec
        self._scenario_id = scenario_id
        self._run_id: ContextVar[str | None] = ContextVar(
            f"rollouteval_external_run_{uuid4().hex}", default=None
        )

    def set_run_id(self, run_id: str | None) -> Token[str | None]:
        return self._run_id.set(run_id)

    def reset_run_id(self, token: Token[str | None]) -> None:
        self._run_id.reset(token)

    def respond(
        self,
        task: AgentTask,
        interactions: tuple[AgentToolInteraction, ...],
    ) -> AgentModelResponse:
        run_id = self._run_id.get()
        request = ExternalAgentRequest(
            protocol_version=self._spec.protocol_version,
            request_id=f"{run_id or 'untraced'}:{len(interactions) + 1}:{uuid4().hex}",
            scenario_id=self._scenario_id,
            run_id=run_id,
            task=task.task,
            interactions=interactions,
        )
        request_bytes = (
            request.model_dump_json(exclude_none=False, by_alias=False).encode("utf-8")
            + b"\n"
        )
        stdout = _run_command(self._spec, request_bytes)
        try:
            value = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExternalAgentMalformedJSONError() from exc
        if not isinstance(value, dict):
            raise InvalidExternalAgentResponseError()
        protocol_version = value.get("protocol_version")
        if protocol_version is None:
            raise InvalidExternalAgentResponseError()
        if type(protocol_version) is not int or protocol_version != self._spec.protocol_version:
            raise UnsupportedExternalAgentProtocolError()
        try:
            return ExternalAgentResponse.model_validate(value).response
        except ValidationError as exc:
            raise InvalidExternalAgentResponseError() from exc


class _RunCorrelatedAgentExecutor(AgentRunExecutor):
    def __init__(self, runner: AgentRunner, model: _SubprocessAgentModel) -> None:
        self._runner = runner
        self._model = model

    def run(
        self,
        task: AgentTask,
        *,
        trace_recorder: TraceRecorder | None = None,
    ):
        run_id = trace_recorder.snapshot().run_id if trace_recorder is not None else None
        token = self._model.set_run_id(run_id)
        try:
            return self._runner.run(task, trace_recorder=trace_recorder)
        finally:
            self._model.reset_run_id(token)


@dataclass(frozen=True)
class SubprocessAgentExecutor:
    """Adapt an ExternalAgentSpec to the existing agent-run executor contract."""

    spec: ExternalAgentSpec

    def create_runner(self, environment: RolloutEnvironment) -> AgentRunExecutor:
        model = _SubprocessAgentModel(self.spec, environment.scenario_id)
        runner = AgentRunner(
            model,
            RolloutObservationTools(environment),
            RolloutMutationTools(environment),
            max_steps=self.spec.max_steps,
        )
        return _RunCorrelatedAgentExecutor(runner, model)


def build_external_agent(spec: ExternalAgentSpec) -> AgentDefinition:
    """Build a benchmark-compatible agent without persisting connection secrets."""
    identity = AgentIdentity(
        agent_id=spec.agent_id,
        version=spec.version,
        implementation_sha256=spec.implementation_sha256,
        connection_type="local_subprocess",
        protocol_version=spec.protocol_version,
    )
    return AgentDefinition(identity=identity, executor=SubprocessAgentExecutor(spec))
