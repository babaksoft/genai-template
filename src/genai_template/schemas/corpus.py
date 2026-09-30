"""Typed results produced by document readers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import overload

from llama_index.core import Document

from genai_template.workflow.portfolio.domain.manifest import NormalizedCorpusManifest


@dataclass(frozen=True)
class LoadedDocuments(Sequence[Document]):
    """Ordered documents and optional validated corpus provenance.

    The sequence methods intentionally preserve the small list-like surface of
    the original reader result while giving the indexing pipeline an explicit
    place to receive corpus provenance.

    Attributes:
        documents:
            Ordered documents loaded from the source.
        provenance:
            Optional normalized provenance for a manifest-backed corpus.
    """

    documents: tuple[Document, ...]
    provenance: NormalizedCorpusManifest | None = None

    def __len__(self) -> int:
        """Return the number of loaded documents.

        Returns:
            Number of documents in the result.
        """

        return len(self.documents)

    def __iter__(self) -> Iterator[Document]:
        """Iterate over loaded documents in source order.

        Returns:
            Iterator over the loaded documents.
        """

        return iter(self.documents)

    @overload
    def __getitem__(self, index: int) -> Document: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Document, ...]: ...

    def __getitem__(self, index: int | slice) -> Document | tuple[Document, ...]:
        """Return one document or a tuple slice.

        Args:
            index:
                Integer index or slice.

        Returns:
            Selected document or tuple of documents.
        """

        return self.documents[index]

    def __eq__(self, other: object) -> bool:
        """Compare loaded documents with another sequence-like value.

        Args:
            other:
                Value to compare.

        Returns:
            Whether document contents and provenance are equal.
        """

        if isinstance(other, LoadedDocuments):
            return (
                self.documents == other.documents
                and self.provenance == other.provenance
            )
        if isinstance(other, (list, tuple)):
            return self.documents == tuple(other)
        return False
