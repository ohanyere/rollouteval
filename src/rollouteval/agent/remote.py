"""Remote HTTP transport for the shared versioned external-agent protocol."""

import socket
import ssl
import urllib.error
import urllib.request
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from rollouteval.agent.contracts import AgentModelResponse, AgentTask, AgentToolInteraction
from rollouteval.agent.external import (
    EXTERNAL_AGENT_PROTOCOL_VERSION,
    ExternalAgentError,
    ExternalAgentRequest,
    UnsupportedExternalAgentProtocolError,
    decode_external_agent_response,
    encode_external_agent_request,
)
from rollouteval.agent.runner import AgentModel, AgentRunExecutor, AgentRunner
from rollouteval.benchmark import AgentDefinition, AgentIdentity
from rollouteval.environment import RolloutEnvironment
from rollouteval.observability.trace import TraceRecorder
from rollouteval.tools import RolloutMutationTools, RolloutObservationTools


class RemoteAgentError(ExternalAgentError):
    """Base safe error for remote agent transport and protocol failures."""


class RemoteAgentConfigurationError(RemoteAgentError):
    """The configured remote endpoint is invalid or unsupported."""


class RemoteAgentConnectionError(RemoteAgentError):
    """DNS or network connection failed."""


class RemoteAgentTLSFailureError(RemoteAgentError):
    """TLS negotiation or certificate validation failed."""


class RemoteAgentConnectTimeoutError(RemoteAgentError):
    """Connection establishment or response-header wait timed out."""


class RemoteAgentReadTimeoutError(RemoteAgentError):
    """Reading the remote response body timed out."""


class RemoteAgentHTTPStatusError(RemoteAgentError):
    """The remote service returned a non-success HTTP status."""


class RemoteAgentContentTypeError(RemoteAgentError):
    """The remote response did not declare a supported JSON content type."""


class RemoteAgentInvalidUTF8Error(RemoteAgentError):
    """The remote response body was not valid UTF-8."""


class RemoteAgentResponseLimitError(RemoteAgentError):
    """The remote response exceeded its configured byte limit."""


class RemoteAgentSpec(BaseModel):
    """Immutable remote endpoint identity and bounded HTTP configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=128)
    implementation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    endpoint_url: str = Field(min_length=1, max_length=2048, repr=False)
    timeout_seconds: float = Field(default=30, gt=0, le=300)
    max_request_bytes: int = Field(default=1_048_576, ge=1, le=8_388_608)
    max_response_bytes: int = Field(default=1_048_576, ge=1, le=8_388_608)
    protocol_version: Literal[1] = EXTERNAL_AGENT_PROTOCOL_VERSION


def _validate_endpoint(endpoint: str) -> str:
    if any(ord(character) < 32 or character.isspace() for character in endpoint):
        raise RemoteAgentConfigurationError()
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError:
        raise RemoteAgentConfigurationError() from None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise RemoteAgentConfigurationError()
    return endpoint


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def _open_remote_request(
    request: urllib.request.Request,
    *,
    timeout: float,
):
    opener = urllib.request.build_opener(_NoRedirectHandler())
    return opener.open(request, timeout=timeout)


def _post_json(
    spec: RemoteAgentSpec,
    payload: bytes,
    bearer_token: SecretStr | None,
) -> bytes:
    endpoint = _validate_endpoint(spec.endpoint_url)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if bearer_token is not None:
        headers["Authorization"] = f"Bearer {bearer_token.get_secret_value()}"
    request = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
    try:
        response = _open_remote_request(request, timeout=spec.timeout_seconds)
    except urllib.error.HTTPError as exc:
        exc.close()
        raise RemoteAgentHTTPStatusError() from None
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ssl.SSLError):
            raise RemoteAgentTLSFailureError() from None
        if isinstance(exc.reason, (socket.timeout, TimeoutError)):
            raise RemoteAgentConnectTimeoutError() from None
        raise RemoteAgentConnectionError() from None
    except ssl.SSLError:
        raise RemoteAgentTLSFailureError() from None
    except (socket.timeout, TimeoutError):
        raise RemoteAgentConnectTimeoutError() from None
    except OSError:
        raise RemoteAgentConnectionError() from None

    with response:
        status = getattr(response, "status", response.getcode())
        if not 200 <= status < 300:
            raise RemoteAgentHTTPStatusError()
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if not (
            content_type == "application/json"
            or (content_type.startswith("application/") and content_type.endswith("+json"))
        ):
            raise RemoteAgentContentTypeError()
        declared_length = response.headers.get("Content-Length")
        if declared_length is not None:
            try:
                if int(declared_length) > spec.max_response_bytes:
                    raise RemoteAgentResponseLimitError()
            except ValueError:
                raise RemoteAgentContentTypeError() from None
        try:
            body = response.read(spec.max_response_bytes + 1)
        except (socket.timeout, TimeoutError):
            raise RemoteAgentReadTimeoutError() from None
        except OSError:
            raise RemoteAgentConnectionError() from None
    if len(body) > spec.max_response_bytes:
        raise RemoteAgentResponseLimitError()
    return body


class _RemoteHTTPAgentModel(AgentModel):
    def __init__(self, spec: RemoteAgentSpec, bearer_token: SecretStr | None, scenario_id: str):
        self._spec = spec
        self._bearer_token = bearer_token
        self._scenario_id = scenario_id
        self._run_id: ContextVar[str | None] = ContextVar(
            f"rollouteval_remote_run_{uuid4().hex}", default=None
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
        payload = encode_external_agent_request(
            request, max_bytes=self._spec.max_request_bytes
        )
        response_body = _post_json(self._spec, payload, self._bearer_token)
        try:
            response_body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RemoteAgentInvalidUTF8Error() from None
        return decode_external_agent_response(
            response_body, protocol_version=self._spec.protocol_version
        )


@dataclass(frozen=True)
class RemoteHTTPAgentExecutor(AgentRunExecutor):
    """Adapt remote HTTP calls to the existing provider-neutral run contract."""

    spec: RemoteAgentSpec
    bearer_token: SecretStr | None = field(default=None, repr=False, compare=False)

    def create_runner(self, environment: RolloutEnvironment) -> AgentRunExecutor:
        model = _RemoteHTTPAgentModel(self.spec, self.bearer_token, environment.scenario_id)
        runner = AgentRunner(
            model,
            RolloutObservationTools(environment),
            RolloutMutationTools(environment),
        )
        return _RunCorrelatedRemoteExecutor(runner, model)


class _RunCorrelatedRemoteExecutor(AgentRunExecutor):
    def __init__(self, runner: AgentRunner, model: _RemoteHTTPAgentModel) -> None:
        self._runner = runner
        self._model = model

    def run(self, task: AgentTask, *, trace_recorder: TraceRecorder | None = None):
        run_id = trace_recorder.snapshot().run_id if trace_recorder is not None else None
        token = self._model.set_run_id(run_id)
        try:
            return self._runner.run(task, trace_recorder=trace_recorder)
        finally:
            self._model.reset_run_id(token)


def build_remote_agent(
    spec: RemoteAgentSpec,
    *,
    bearer_token: str | SecretStr | None = None,
) -> AgentDefinition:
    """Create an agent definition; bearer credentials stay runtime-only."""
    secret = (
        bearer_token
        if isinstance(bearer_token, SecretStr) or bearer_token is None
        else SecretStr(bearer_token)
    )
    identity = AgentIdentity(
        agent_id=spec.agent_id,
        version=spec.version,
        implementation_sha256=spec.implementation_sha256,
        connection_type="remote_http",
        protocol_version=spec.protocol_version,
    )
    return AgentDefinition(
        identity=identity,
        executor=RemoteHTTPAgentExecutor(spec=spec, bearer_token=secret),
    )
