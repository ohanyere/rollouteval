from types import SimpleNamespace

import pytest

from rollouteval.provider_client import (
    GROQ_BASE_URL,
    configured_api_provider,
    create_api_client,
)


def test_groq_client_uses_openai_sdk_compatible_base_url_and_key(monkeypatch) -> None:
    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(responses=object())

    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr("openai.OpenAI", fake_openai)

    client, provider = create_api_client("groq")

    assert provider == "groq"
    assert client.responses is not None
    assert captured == {"api_key": "test-groq-key", "base_url": GROQ_BASE_URL}


@pytest.mark.parametrize(
    ("provider", "key_name"),
    [("groq", "GROQ_API_KEY"), ("openai", "OPENAI_API_KEY")],
)
def test_explicit_provider_requires_its_own_api_key(
    monkeypatch, provider: str, key_name: str
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    alternate_key = "OPENAI_API_KEY" if provider == "groq" else "GROQ_API_KEY"
    monkeypatch.setenv(alternate_key, "alternate-test-key")

    assert configured_api_provider(provider) is None
    with pytest.raises(RuntimeError, match=key_name):
        create_api_client(provider)


def test_openai_remains_default_when_both_provider_keys_are_configured(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    assert configured_api_provider() == "openai"


def test_openai_client_keeps_default_sdk_configuration(monkeypatch) -> None:
    captured = []

    def fake_openai(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(responses=object())

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr("openai.OpenAI", fake_openai)

    client, provider = create_api_client("openai")

    assert provider == "openai"
    assert client.responses is not None
    assert captured == [{}]


def test_provider_selection_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="provider must be"):
        configured_api_provider("other")
