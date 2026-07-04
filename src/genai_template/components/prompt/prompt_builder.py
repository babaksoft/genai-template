from genai_template.prompts.rag_templates import SINGLE_TURN_QA

"""Prompt builder component."""


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

        return SINGLE_TURN_QA.format(query=query, context=context)
