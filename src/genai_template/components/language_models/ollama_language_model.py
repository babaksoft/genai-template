"""Ollama language model adapter."""

import logging

from llama_index.llms.ollama import Ollama

from genai_template.config import settings
from genai_template.config.ollama import resolve_ollama_base_url
from genai_template.utils import Timer

logger = logging.getLogger(__name__)


class OllamaLanguageModel:
    """Language model adapter backed by Ollama."""

    def __init__(
        self,
        model_name: str = settings.LLM_MODEL,
        base_url: str | None = None,
        request_timeout: float = settings.REQUEST_TIMEOUT,
    ) -> None:
        """
        Initialize the language model.

        Args:
            model_name:
                Name of the Ollama model.
            base_url:
                Ollama server URL. When omitted, resolve the application
                default for the current environment.
            request_timeout:
                Maximum number of seconds to wait for an Ollama request.
        """

        self._model_name = model_name
        self._llm = Ollama(
            model=model_name,
            base_url=base_url or resolve_ollama_base_url(),
            request_timeout=request_timeout,
        )

    def generate(self, prompt: str) -> str:
        """
        Generate a response for a prompt.

        Args:
            prompt:
                Prompt to send to the language model.

        Returns:
            Generated response.
        """

        logger.info("Generating response using model '%s'.", self._model_name)
        logger.info("Prompt length: %d characters", len(prompt))

        with Timer() as timer:
            response = self._llm.complete(prompt)

        logger.info("Response generation completed in %.3f second(s).", timer.elapsed)
        logger.info("Response length: %d characters", len(str(response.text)))

        return str(response.text)
