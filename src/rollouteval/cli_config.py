"""Safe local configuration for the RolloutEval command-line interface."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rollouteval.benchmark import BenchmarkSpec


class CLIConfigurationError(Exception):
    """A safe, user-facing configuration or catalog error."""


class AgentConfiguration(BaseModel):
    """Persisted agent connection settings; secret values are never accepted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    agent_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=128)
    implementation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    connection_type: Literal["local_subprocess", "remote_http"]
    protocol_version: Literal[1] = 1
    command: tuple[str, ...] | None = None
    working_directory: str | None = None
    allowed_environment_names: tuple[str, ...] = ()
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    max_request_bytes: int = Field(default=1_048_576, ge=1, le=8_388_608)
    max_stdout_bytes: int = Field(default=1_048_576, ge=1, le=8_388_608)
    max_stderr_bytes: int = Field(default=65_536, ge=1, le=1_048_576)
    max_response_bytes: int = Field(default=1_048_576, ge=1, le=8_388_608)
    max_steps: int = Field(default=8, ge=1, le=64)
    endpoint_url: str | None = None
    token_environment_name: str | None = None

    @model_validator(mode="after")
    def validate_connection_fields(self) -> "AgentConfiguration":
        if self.connection_type == "local_subprocess":
            if not self.command or self.endpoint_url is not None:
                raise ValueError("subprocess configuration requires argv only")
            if self.token_environment_name is not None:
                raise ValueError("subprocess configuration cannot contain a token reference")
        else:
            if self.command is not None or self.endpoint_url is None:
                raise ValueError("remote configuration requires an endpoint only")
            if self.working_directory is not None or self.allowed_environment_names:
                raise ValueError("remote configuration cannot contain subprocess settings")
            _validate_remote_endpoint(self.endpoint_url)
        names = self.allowed_environment_names
        if len(names) != len(set(names)) or any(not _valid_environment_name(name) for name in names):
            raise ValueError("environment references must be unique valid names")
        if self.token_environment_name is not None and not _valid_environment_name(
            self.token_environment_name
        ):
            raise ValueError("token environment reference is invalid")
        if self.command and any(not arg or "\x00" in arg for arg in self.command):
            raise ValueError("command argv contains an invalid argument")
        if self.command:
            secret_option = re.compile(r"(?i)(?:token|secret|password|authorization|api[-_]?key)")
            secret_value = re.compile(r"(?i)(?:\bbearer\s+\S+|\b(?:sk|gsk)[_-][A-Za-z0-9_-]{8,})")
            for index, argument in enumerate(self.command):
                if secret_value.search(argument) or (
                    secret_option.search(argument.lstrip("-"))
                    and argument.startswith("-")
                    and index + 1 < len(self.command)
                    and not self.command[index + 1].startswith("-")
                ):
                    raise ValueError("credentials must be supplied through environment references")
        if self.endpoint_url and _contains_url_credentials(self.endpoint_url):
            raise ValueError("endpoint URL must not contain credentials")
        return self


class AgentConfigurationEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    configuration: AgentConfiguration


class BenchmarkConfigurationEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    benchmark: BenchmarkSpec


def default_config_dir() -> Path:
    """Return the conventional per-user XDG config location."""
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "rollouteval"


def default_artifact_dir() -> Path:
    """Return the conventional per-user XDG data location for run artifacts."""
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "rollouteval" / "artifacts"


def _valid_environment_name(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", name))


def _contains_url_credentials(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        return parsed.username is not None or parsed.password is not None
    except ValueError:
        return True


def _validate_remote_endpoint(url: str) -> None:
    if any(ord(character) < 32 or character.isspace() for character in url):
        raise ValueError("endpoint URL is malformed")
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        raise ValueError("endpoint URL is malformed") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ValueError("endpoint must be HTTP(S) without credentials, query, or fragment")


def _filename(prefix: str, identity: str) -> str:
    if not identity or len(identity) > 128:
        raise CLIConfigurationError("invalid catalog identity")
    return f"{prefix}-{hashlib.sha256(identity.encode('utf-8')).hexdigest()}.json"


class CLIConfigurationStore:
    """Versioned local agent/benchmark catalogs with atomic publication."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()

    @property
    def agents_root(self) -> Path:
        return self.root / "agents"

    @property
    def benchmarks_root(self) -> Path:
        return self.root / "benchmarks"

    def _save_json(self, path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                raise CLIConfigurationError("catalog path is not a regular file")
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                raise CLIConfigurationError("existing catalog entry is corrupt") from None
            if existing == value:
                return
            raise CLIConfigurationError("catalog identity already exists with different content")
        descriptor, temporary_name = tempfile.mkstemp(prefix=".rollouteval-", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise CLIConfigurationError("catalog identity already exists") from None
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _read(path: Path) -> dict:
        if path.is_symlink() or not path.is_file():
            raise CLIConfigurationError("catalog entry is not a regular file")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise CLIConfigurationError("catalog entry is malformed JSON") from None
        if not isinstance(value, dict):
            raise CLIConfigurationError("catalog entry must be a JSON object")
        return value

    def save_agent(self, configuration: AgentConfiguration) -> Path:
        config = AgentConfiguration.model_validate(configuration.model_dump(mode="python"))
        path = self.agents_root / _filename("agent", config.agent_id)
        self._save_json(path, AgentConfigurationEnvelope(configuration=config).model_dump(mode="json"))
        return path

    def load_agent(self, agent_id: str) -> AgentConfiguration:
        path = self.agents_root / _filename("agent", agent_id)
        if not path.exists() and not path.is_symlink():
            raise FileNotFoundError("configured agent does not exist")
        try:
            return AgentConfigurationEnvelope.model_validate(self._read(path)).configuration
        except ValidationError:
            raise CLIConfigurationError("agent configuration is invalid") from None

    def list_agents(self) -> tuple[AgentConfiguration, ...]:
        if not self.agents_root.exists():
            return ()
        configs = []
        for path in sorted(self.agents_root.glob("agent-*.json")):
            try:
                config = AgentConfigurationEnvelope.model_validate(self._read(path)).configuration
            except ValidationError:
                raise CLIConfigurationError("agent configuration is invalid") from None
            if path.name != _filename("agent", config.agent_id):
                raise CLIConfigurationError("agent catalog identity mismatch")
            configs.append(config)
        return tuple(sorted(configs, key=lambda item: item.agent_id))

    def remove_agent(self, agent_id: str) -> None:
        path = self.agents_root / _filename("agent", agent_id)
        if path.is_symlink():
            raise CLIConfigurationError("agent configuration path is a symlink")
        if not path.is_file():
            raise FileNotFoundError("configured agent does not exist")
        path.unlink()

    def save_benchmark(self, benchmark: BenchmarkSpec) -> Path:
        path = self.benchmarks_root / _filename("benchmark", benchmark.benchmark_id)
        envelope = BenchmarkConfigurationEnvelope(benchmark=benchmark)
        self._save_json(path, envelope.model_dump(mode="json"))
        return path

    def load_benchmark(self, benchmark_id: str) -> BenchmarkSpec:
        path = self.benchmarks_root / _filename("benchmark", benchmark_id)
        if not path.exists() and not path.is_symlink():
            raise FileNotFoundError("configured benchmark does not exist")
        try:
            return BenchmarkConfigurationEnvelope.model_validate(self._read(path)).benchmark
        except ValidationError:
            raise CLIConfigurationError("benchmark configuration is invalid") from None

    def list_benchmarks(self) -> tuple[BenchmarkSpec, ...]:
        if not self.benchmarks_root.exists():
            return ()
        benchmarks = []
        for path in sorted(self.benchmarks_root.glob("benchmark-*.json")):
            try:
                benchmark = BenchmarkConfigurationEnvelope.model_validate(self._read(path)).benchmark
            except ValidationError:
                raise CLIConfigurationError("benchmark configuration is invalid") from None
            if path.name != _filename("benchmark", benchmark.benchmark_id):
                raise CLIConfigurationError("benchmark catalog identity mismatch")
            benchmarks.append(benchmark)
        return tuple(sorted(benchmarks, key=lambda item: item.benchmark_id))
