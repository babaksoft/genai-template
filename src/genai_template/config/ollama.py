"""Ollama endpoint selection helpers."""

import ipaddress
import logging
import os
import subprocess
from functools import lru_cache

import httpx

from genai_template.config import settings

logger = logging.getLogger(__name__)

_LOCAL_OLLAMA_URL = settings.OLLAMA_BASE_URL
_PROBE_TIMEOUT_SECONDS = 0.5


def _is_wsl() -> bool:
    """Return whether this process is running inside WSL."""

    return bool(os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"))


def _is_ollama_available(base_url: str) -> bool:
    """Return whether an Ollama server responds at ``base_url``."""

    try:
        response = httpx.get(
            f"{base_url.rstrip('/')}/api/tags",
            timeout=_PROBE_TIMEOUT_SECONDS,
            trust_env=False,
        )
    except httpx.HTTPError:
        return False

    return response.is_success


def _windows_host_gateway() -> str | None:
    """Return the Windows-host gateway address exposed to WSL NAT."""

    try:
        route = subprocess.run(
            ["ip", "-4", "route", "show", "default"],
            capture_output=True,
            check=False,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    for line in route.stdout.splitlines():
        fields = line.split()
        if "via" not in fields:
            continue

        gateway_index = fields.index("via") + 1
        if gateway_index == len(fields):
            continue
        gateway = fields[gateway_index]
        try:
            address = ipaddress.ip_address(gateway)
        except ValueError:
            continue
        if address.version == 4:
            return str(address)

    return None


@lru_cache(maxsize=1)
def resolve_ollama_base_url() -> str:
    """Resolve the Ollama endpoint for local and WSL NAT execution.

    ``OLLAMA_BASE_URL`` is an explicit user choice and always takes precedence.
    Without it, localhost is preferred and WSL NAT falls back to the current
    Windows-host gateway when no local Ollama server is available.
    """

    configured_url = os.environ.get("OLLAMA_BASE_URL")
    if configured_url:
        return configured_url

    if _is_ollama_available(_LOCAL_OLLAMA_URL):
        return _LOCAL_OLLAMA_URL

    if _is_wsl():
        gateway = _windows_host_gateway()
        if gateway:
            base_url = f"http://{gateway}:11434"
            logger.info("Using the WSL NAT Windows-host Ollama endpoint: %s", base_url)
            return base_url

        logger.warning(
            "Unable to determine the Windows-host gateway for Ollama; "
            "falling back to %s. Set OLLAMA_BASE_URL to override it.",
            _LOCAL_OLLAMA_URL,
        )

    return _LOCAL_OLLAMA_URL
