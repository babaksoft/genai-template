"""OpenAI language model adapter."""

import logging

from llama_index.llms.openai import OpenAI

from genai_template.config import settings
from genai_template.utils import Timer

logger = logging.getLogger(__name__)


class OpenAILanguageModel:
    """Language model adapter backed by OpenAI."""

    def __init__(
        self,
        model_name: str,
        request_timeout: float = settings.REQUEST_TIMEOUT,
    ) -> None:
        """Initialize the OpenAI language model.

        Args:
            model_name:
                Name of the OpenAI language model.
            request_timeout:
                Maximum number of seconds to wait for an OpenAI request.
        """

        self._model_name = model_name
        self._llm = OpenAI(model=model_name, timeout=request_timeout)

    def generate(self, prompt: str) -> str:
        """Generate a response for a prompt.

        Args:
            prompt:
                Prompt to send to the language model.

        Returns:
            Generated response text.
        """

        logger.info("Generating response using model '%s'.", self._model_name)
        logger.info("Prompt length: %d characters", len(prompt))

        with Timer() as timer:
            response = self._llm.complete(prompt)

        text = str(response.text)
        logger.info("Response generation completed in %.3f second(s).", timer.elapsed)
        logger.info("Response length: %d characters", len(text))

        return text
