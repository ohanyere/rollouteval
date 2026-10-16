"""Groq Responses API implementation of the groundedness judge interface."""

import hashlib
from typing import Any

from rollouteval.evaluation.groundedness import GroundednessJudgment
from rollouteval.evaluation.grounding import GroundingContext
from rollouteval.evaluation.providers.openai import (
    GROUNDEDNESS_INSTRUCTIONS,
    GroundednessJudgeResponseError,
    OpenAIGroundednessJudge,
)
from rollouteval.provider_client import create_api_client


class _GroqResponsesClient:
    def __init__(self, client: Any) -> None:
        self._client = client
        self.responses = self

    def create(self, **request: Any) -> Any:
        # Groq's Responses endpoint does not accept the OpenAI-only store option.
        request.pop("store", None)
        return self._client.responses.create(**request)


class GroqGroundednessJudge:
    """Judge groundedness through Groq without sending unsupported store."""

    def __init__(self, *, model: str, client: Any | None = None) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if client is None:
            client, _ = create_api_client("groq")
        self.model = model.strip()
        self.provider_name = "groq"
        self.prompt_sha256 = hashlib.sha256(
            GROUNDEDNESS_INSTRUCTIONS.encode()
        ).hexdigest()
        self._delegate = OpenAIGroundednessJudge(
            model=model,
            client=_GroqResponsesClient(client),
        )

    def judge(self, context: GroundingContext) -> GroundednessJudgment:
        try:
            return self._delegate.judge(context)
        except GroundednessJudgeResponseError as exc:
            raise GroundednessJudgeResponseError(
                str(exc).replace("OpenAI", "Groq")
            ) from exc
