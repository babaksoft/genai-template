"""Unit tests for the text reader."""

from pathlib import Path

import pytest
from llama_index.core import Document

from genai_template.components.readers import TextReader


def test_load_documents(tmp_path: Path) -> None:
    """Reader should load Markdown and text documents."""

    markdown = tmp_path / "document.md"
    markdown.write_text(
        "# Title\n\nHello Markdown.",
        encoding="utf-8",
    )

    text = tmp_path / "notes.txt"
    text.write_text(
        "Hello text.",
        encoding="utf-8",
    )

    reader = TextReader()

    documents = reader.load(tmp_path)

    assert len(documents) == 2
    assert all(isinstance(doc, Document) for doc in documents)


def test_empty_directory(tmp_path: Path) -> None:
    """Reader should return an empty list."""

    reader = TextReader()

    documents = reader.load(tmp_path)

    assert documents == []


def test_missing_directory() -> None:
    """Loading a missing directory should fail."""

    reader = TextReader()

    with pytest.raises(FileNotFoundError):
        reader.load(Path("does_not_exist"))


def test_path_is_not_directory(tmp_path: Path) -> None:
    """Loading a file instead of a directory should fail."""

    file_path = tmp_path / "document.txt"
    file_path.write_text(
        "Hello",
        encoding="utf-8",
    )

    reader = TextReader()

    with pytest.raises(NotADirectoryError):
        reader.load(file_path)
