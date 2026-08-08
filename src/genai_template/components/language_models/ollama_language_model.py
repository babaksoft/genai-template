"""Ollama language model adapter."""

import logging

from llama_index.llms.ollama import Ollama

from genai_template.utils import Timer

logger = logging.getLogger(__name__)


class OllamaLanguageModel:
    """Language model adapter backed by Ollama."""

    def __init__(self, model_name: str) -> None:
        """
        Initialize the language model.

        Args:
            model_name:
                Name of the Ollama model.
        """

        self._model_name = model_name
        self._llm = Ollama(
            model=model_name,
            request_timeout=180,
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
