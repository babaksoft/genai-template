"""Prompt builder component."""

import logging

from genai_template.observability import INPUT_VALUE, OUTPUT_VALUE, application_span
from genai_template.prompts.rag_templates import SINGLE_TURN_QA
from genai_template.utils import Timer

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

        with application_span(
            "rag.prompt.build",
            "PROMPT",
            {INPUT_VALUE: query, "rag.context": context},
        ) as span:
            with Timer() as timer:
                prompt = SINGLE_TURN_QA.format(query=query, context=context)
            span.set_attribute(OUTPUT_VALUE, prompt)

        logger.info("Built prompt in %.3f second(s).", timer.elapsed)
        logger.info("Prompt length: %d characters", len(prompt))

        return prompt
