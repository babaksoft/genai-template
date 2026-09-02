"""Tests for Ollama endpoint selection."""

from unittest.mock import MagicMock, patch

from pytest import MonkeyPatch

from genai_template.config import ollama


def _resolve() -> str:
    ollama.resolve_ollama_base_url.cache_clear()
    return ollama.resolve_ollama_base_url()


def test_explicit_url_takes_precedence(monkeypatch: MonkeyPatch) -> None:
    """An explicit endpoint bypasses endpoint discovery."""

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://configured:11434")

    with patch.object(ollama, "_is_ollama_available") as mock_available:
        assert _resolve() == "http://configured:11434"

    mock_available.assert_not_called()


@patch.object(ollama, "_is_ollama_available", return_value=True)
def test_reachable_localhost_is_used(
    mock_available: MagicMock, monkeypatch: MonkeyPatch
) -> None:
    """A local Ollama endpoint wins over the WSL gateway."""

    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    assert _resolve() == "http://localhost:11434"
    mock_available.assert_called_once_with("http://localhost:11434")


@patch.object(ollama, "_windows_host_gateway", return_value="172.31.80.1")
@patch.object(ollama, "_is_wsl", return_value=True)
@patch.object(ollama, "_is_ollama_available", return_value=False)
def test_wsl_uses_current_default_gateway(
    mock_available: MagicMock,
    mock_is_wsl: MagicMock,
    mock_gateway: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    """WSL NAT uses its current Windows-host gateway when localhost fails."""

    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    assert _resolve() == "http://172.31.80.1:11434"
    mock_available.assert_called_once_with("http://localhost:11434")
    mock_is_wsl.assert_called_once_with()
    mock_gateway.assert_called_once_with()


@patch.object(ollama, "_windows_host_gateway", return_value=None)
@patch.object(ollama, "_is_wsl", return_value=True)
@patch.object(ollama, "_is_ollama_available", return_value=False)
def test_missing_wsl_gateway_falls_back_to_localhost(
    mock_available: MagicMock,
    mock_is_wsl: MagicMock,
    mock_gateway: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    """Unavailable WSL routing preserves the conventional local endpoint."""

    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    assert _resolve() == "http://localhost:11434"
    mock_available.assert_called_once_with("http://localhost:11434")
    mock_is_wsl.assert_called_once_with()
    mock_gateway.assert_called_once_with()
