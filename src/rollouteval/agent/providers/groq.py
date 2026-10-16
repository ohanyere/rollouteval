"""Groq Responses API implementation of the provider-neutral agent model."""

from typing import Any

from rollouteval.agent.contracts import AgentModelResponse, AgentTask, AgentToolInteraction
from rollouteval.agent.providers.openai import (
    AgentProviderResponseError,
    OpenAIResponsesModel,
)
from rollouteval.provider_client import create_api_client


class GroqResponsesModel:
    """Implement AgentModel through Groq's OpenAI-compatible Responses API."""

    def __init__(self, model: str, client: Any | None = None) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if client is None:
            client, _ = create_api_client("groq")
        self._delegate = OpenAIResponsesModel(model=model, client=client)

    def respond(
        self,
        task: AgentTask,
        interactions: tuple[AgentToolInteraction, ...],
    ) -> AgentModelResponse:
        try:
            return self._delegate.respond(task, interactions)
        except AgentProviderResponseError as exc:
            raise AgentProviderResponseError(str(exc).replace("OpenAI", "Groq")) from exc
