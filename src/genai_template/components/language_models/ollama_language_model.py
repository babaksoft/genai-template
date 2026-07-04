"""Ollama language model adapter."""

from llama_index.llms.ollama import Ollama


class OllamaLanguageModel:
    """Language model adapter backed by Ollama."""

    def __init__(self, model_name: str) -> None:
        """
        Initialize the language model.

        Args:
            model_name:
                Name of the Ollama model.
        """

        self._llm = Ollama(model=model_name)

    def generate(self, prompt: str) -> str:
        """
        Generate a response for a prompt.

        Args:
            prompt:
                Prompt to send to the language model.

        Returns:
            Generated response.
        """

        response = self._llm.complete(prompt)

        return str(response.text)
