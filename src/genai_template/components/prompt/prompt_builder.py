"""Prompt builder component."""

import logging

from genai_template.prompts.rag_templates import SINGLE_TURN_QA
from genai_template.utils.timer import Timer

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Build prompts for retrieval-augmented generation."""

    def build(
        self,
        query: str,
        context: str,
    ) -> str:
        """Build an LLM prompt.

        Args:
            query:
                User query.

            context:
                Retrieved context.

        Returns:
            Prompt ready to be sent to an LLM.
        """

        logger.info(
            "Input length (characters): query=%d context=%d",
            len(query),
            len(context),
        )

        with Timer() as timer:
            prompt = SINGLE_TURN_QA.format(query=query, context=context)

        logger.info("Built prompt in %.3f second(s).", timer.elapsed)
        logger.info("Prompt length: %d characters", len(prompt))

        return prompt
