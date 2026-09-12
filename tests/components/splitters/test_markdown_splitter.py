"""Unit tests for the Markdown document splitter."""

from unittest.mock import MagicMock, patch

from llama_index.core import Document

from genai_template.components.splitters import MarkdownDocumentSplitter


@patch("genai_template.components.splitters.markdown_splitter.MarkdownNodeParser")
def test_constructor_forwards_header_path_separator(
    mock_parser: MagicMock,
) -> None:
    """The configured separator should reach the underlying parser."""

    MarkdownDocumentSplitter(header_path_separator=" > ")

    mock_parser.assert_called_once_with(header_path_separator=" > ")


def test_header_hierarchy_and_source_metadata_are_preserved() -> None:
    """Chunks should retain source fields and parser-generated header paths."""

    document = Document(
        text="# Guide\nIntro.\n## Install\nInstall details.\n### Linux\nLinux details.",
        metadata={"file_name": "guide.md", "author": "Babak"},
    )

    chunks = MarkdownDocumentSplitter().split([document])

    assert [chunk.metadata["header_path"] for chunk in chunks] == [
        "/",
        "/Guide/",
        "/Guide/Install/",
    ]
    assert all(chunk.metadata["author"] == "Babak" for chunk in chunks)
    assert all(chunk.document_id == "guide.md" for chunk in chunks)


def test_custom_header_path_separator() -> None:
    """The parser should construct header paths with the configured separator."""

    document = Document(
        text="# Guide\nIntro.\n## Install\nInstall details.",
        metadata={"file_name": "guide.md"},
    )

    chunks = MarkdownDocumentSplitter(header_path_separator=" > ").split([document])

    assert chunks[1].metadata["header_path"] == " > Guide > "


def test_heading_inside_fenced_code_does_not_split() -> None:
    """Markdown-looking lines in fenced code should remain in their section."""

    document = Document(
        text=(
            "# Example\nBefore code.\n```markdown\n# Not a heading\n```\n"
            "After code.\n## Real heading\nReal section."
        ),
        metadata={"file_name": "example.md"},
    )

    chunks = MarkdownDocumentSplitter().split([document])

    assert len(chunks) == 2
    assert "# Not a heading" in chunks[0].text
    assert chunks[1].metadata["header_path"] == "/Example/"


def test_headerless_document_remains_one_chunk() -> None:
    """A document without Markdown headers should remain a single section."""

    document = Document(
        text="Paragraph one.\n\nParagraph two.",
        metadata={"file_name": "notes.txt"},
    )

    chunks = MarkdownDocumentSplitter().split([document])

    assert len(chunks) == 1
    assert chunks[0].text == "Paragraph one.\n\nParagraph two."
    assert chunks[0].metadata["header_path"] == "/"


def test_chunk_ids_are_deterministic_per_document() -> None:
    """Chunk identifiers should use the existing document-index convention."""

    documents = [
        Document(
            text="# One\nFirst.\n## Two\nSecond.",
            metadata={"file_name": "first.md"},
        ),
        Document(
            text="# Other\nThird.",
            metadata={"file_name": "second.md"},
        ),
    ]

    chunks = MarkdownDocumentSplitter().split(documents)

    assert [chunk.id for chunk in chunks] == [
        "first.md-000",
        "first.md-001",
        "second.md-000",
    ]
    assert all(chunk.embedding is None for chunk in chunks)


def test_empty_input_returns_empty_list() -> None:
    """An empty document list should produce no chunks."""

    assert MarkdownDocumentSplitter().split([]) == []


def test_empty_document_returns_empty_list() -> None:
    """A document without text should produce no chunks."""

    document = Document(text="", metadata={"file_name": "empty.md"})

    assert MarkdownDocumentSplitter().split([document]) == []
