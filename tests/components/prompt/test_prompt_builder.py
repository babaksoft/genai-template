"""Tests for the PromptBuilder."""

from genai_template.components.prompt import PromptBuilder


def test_build_includes_query() -> None:
    """The user query should appear in the generated prompt."""

    builder = PromptBuilder()
    prompt = builder.build(
        query="What is RAG?",
        context="Chunk 1\n\nRetrieval-Augmented Generation.",
    )

    assert "What is RAG?" in prompt


def test_build_includes_context() -> None:
    """The retrieved context should appear in the generated prompt."""

    builder = PromptBuilder()
    prompt = builder.build(
        query="Question",
        context="Chunk 1\n\nImportant context",
    )

    assert "Chunk 1" in prompt
    assert "Important context" in prompt


def test_build_handles_empty_context() -> None:
    """An empty context should still produce a valid prompt."""

    builder = PromptBuilder()
    prompt = builder.build(
        query="Question",
        context="",
    )

    assert "Question" in prompt
    assert "Context:" in prompt
    assert prompt.endswith("Answer:")


def test_build_includes_instructions() -> None:
    """The prompt should contain the instruction block."""

    builder = PromptBuilder()
    prompt = builder.build(
        query="Question",
        context="Context",
    )

    assert "Use the provided context" in prompt
    assert "If the answer cannot be found in the context" in prompt
    assert prompt.endswith("Answer:")
