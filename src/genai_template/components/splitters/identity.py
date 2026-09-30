"""Portable document and chunk identity helpers."""

from __future__ import annotations

from typing import Any

from llama_index.core import Document
from llama_index.core.schema import BaseNode

from genai_template.components.readers.portfolio_metadata import (
    FILE_NAME,
    has_portfolio_metadata,
    validate_portfolio_metadata,
)
from genai_template.schemas import DocumentChunk


def document_identity_map(documents: list[Document]) -> dict[str, str]:
    """Validate documents and map parser source IDs to canonical IDs.

    Args:
        documents:
            Documents about to be split.

    Returns:
        Parser source IDs mapped to portable canonical document IDs.

    Raises:
        ValueError:
            If a document has invalid Portfolio metadata or a duplicate identity.
    """

    identities: dict[str, str] = {}
    seen: set[str] = set()
    for document in documents:
        metadata = dict(document.metadata)
        filename = metadata.get(FILE_NAME)
        if has_portfolio_metadata(metadata):
            validate_portfolio_metadata(metadata, allow_header_path=False)
            if document.id_ != filename:
                raise ValueError("Portfolio document ID must equal its file_name")
            document_id = str(document.id_)
        elif isinstance(filename, str) and filename:
            document_id = filename
        else:
            document_id = str(document.id_)
        if not document_id:
            raise ValueError("Document ID cannot be empty")
        if document_id in seen:
            raise ValueError(f"Duplicate document ID: {document_id}")
        seen.add(document_id)
        identities[str(document.id_)] = document_id
    return identities


def node_document_id(node: BaseNode, identities: dict[str, str]) -> str:
    """Resolve a parser node to its canonical parent document identity.

    Args:
        node:
            Parser-produced node.
        identities:
            Source-to-canonical identity mapping.

    Returns:
        Canonical parent document identity.

    Raises:
        ValueError:
            If the node cannot be tied to exactly one input document.
    """

    source_id = node.ref_doc_id
    if source_id is not None and source_id in identities:
        return identities[source_id]
    filename: Any = node.metadata.get(FILE_NAME)
    matches = [identity for identity in identities.values() if identity == filename]
    if len(matches) == 1:
        return matches[0]
    raise ValueError("Splitter produced a node without a known document identity")


def validate_chunk_metadata(metadata: dict[str, Any], document_id: str) -> None:
    """Ensure Portfolio provenance survives splitting unchanged.

    Args:
        metadata:
            Parser-produced chunk metadata.
        document_id:
            Canonical parent document identity.

    Raises:
        ValueError:
            If Portfolio provenance is incomplete or contradicts the identity.
    """

    if not has_portfolio_metadata(metadata):
        return
    validate_portfolio_metadata(metadata, allow_header_path=True)
    if metadata[FILE_NAME] != document_id:
        raise ValueError("Chunk file_name does not match its document ID")


def validate_unique_chunk_ids(chunks: list[DocumentChunk]) -> None:
    """Reject duplicate canonical chunk identities.

    Args:
        chunks:
            Chunks to validate.

    Raises:
        ValueError:
            If two chunks share an identity.
    """

    ids = [chunk.id for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs are not allowed")
