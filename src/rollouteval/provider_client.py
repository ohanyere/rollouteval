"""Provider client configuration for OpenAI-compatible APIs."""

import os
from typing import Any


GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def configured_api_provider(provider: str | None = None) -> str | None:
    """Return the configured provider, preserving OpenAI as the default."""
    if provider not in (None, "groq", "openai"):
        raise ValueError("provider must be 'groq' or 'openai'")
    if provider is not None:
        key_name = "GROQ_API_KEY" if provider == "groq" else "OPENAI_API_KEY"
        return provider if os.environ.get(key_name) else None
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    return None


def create_api_client(provider: str | None = None) -> tuple[Any, str]:
    """Create an OpenAI SDK client for the configured provider."""
    configured = configured_api_provider(provider)
    if configured is None:
        key_name = "GROQ_API_KEY" if provider == "groq" else "OPENAI_API_KEY"
        if provider is None:
            raise RuntimeError("GROQ_API_KEY or OPENAI_API_KEY must be configured")
        raise RuntimeError(f"{key_name} must be configured for provider {provider!r}")

    from openai import OpenAI

    if configured == "groq":
        return (
            OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url=GROQ_BASE_URL),
            configured,
        )
    return OpenAI(), configured
