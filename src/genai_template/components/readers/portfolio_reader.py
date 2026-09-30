"""Manifest-aware reader for validated Portfolio corpora."""

from __future__ import annotations

from pathlib import Path

from llama_index.core import Document

from genai_template.components.readers.portfolio_metadata import (
    COMPONENT_ID,
    CORPUS_FINGERPRINT,
    DOCUMENT_TYPE,
    FILE_NAME,
    GENERATION_FINGERPRINT,
    PROJECT_DISPLAY_NAME,
    PROJECT_SLUG,
    REPOSITORY_URL,
    RESOLVED_COMMIT_SHA,
    validate_portfolio_metadata,
)
from genai_template.components.readers.text_reader import TextReader
from genai_template.schemas.corpus import LoadedDocuments
from genai_template.workflow.portfolio.corpus.loading import load_corpus
from genai_template.workflow.portfolio.corpus.validation import CorpusValidationError
from genai_template.workflow.portfolio.domain import (
    ManifestDocumentV2,
    ManifestProject,
)


class PortfolioReader:
    """Load validated Portfolio Markdown with controlled stable provenance."""

    def __init__(self, text_reader: TextReader | None = None) -> None:
        """Initialize the Portfolio reader.

        Args:
            text_reader:
                Generic flat-file reader used after manifest validation.
        """

        self._text_reader = text_reader or TextReader(exclude_hidden=False)

    def load(self, directory: Path) -> LoadedDocuments:
        """Load one manifest-backed Portfolio corpus in manifest order.

        Args:
            directory:
                Publication pointer or immutable Portfolio release directory.

        Returns:
            Ordered documents with normalized corpus provenance.

        Raises:
            CorpusValidationError:
                If validation or the generic reader result disagrees with the
                pinned manifest-backed release.
        """

        corpus = load_corpus(directory)
        loaded = self._text_reader.load(corpus.release_path)
        documents = loaded.documents if isinstance(loaded, LoadedDocuments) else loaded
        by_name: dict[str, Document] = {}
        expected_names = {record.filename for record in corpus.documents}

        for document in documents:
            filename = document.metadata.get(FILE_NAME)
            file_path = document.metadata.get("file_path")
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise CorpusValidationError("reader returned an unsafe document name")
            if filename not in expected_names:
                raise CorpusValidationError(
                    "reader returned an extra or renamed document"
                )
            if filename in by_name:
                raise CorpusValidationError("reader returned a duplicate document")
            if not isinstance(file_path, str):
                raise CorpusValidationError("reader omitted the document path")

            try:
                reader_path = Path(file_path).resolve(strict=True)
            except OSError as exc:
                raise CorpusValidationError(
                    "reader returned an invalid document path"
                ) from exc
            if reader_path != corpus.release_path / filename:
                raise CorpusValidationError(
                    "reader returned a document from another path"
                )
            by_name[filename] = document

        if set(by_name) != expected_names:
            raise CorpusValidationError("reader result does not match the manifest")

        projects = {
            project.project_slug: project for project in corpus.manifest.projects
        }
        projected = tuple(
            self._project_document(
                by_name[record.filename],
                record,
                projects[record.project_slug],
                corpus.manifest.corpus_fingerprint,
            )
            for record in corpus.documents
        )
        return LoadedDocuments(documents=projected, provenance=corpus.manifest)

    @staticmethod
    def _project_document(
        document: Document,
        record: ManifestDocumentV2,
        project: ManifestProject,
        corpus_fingerprint: str,
    ) -> Document:
        """Replace reader metadata and identity with the manifest projection.

        Args:
            document:
                Reader-produced document whose text is retained.
            record:
                Owning manifest document record.
            project:
                Owning project provenance.
            corpus_fingerprint:
                Identity of the complete published corpus.

        Returns:
            Fresh document with portable identity and controlled metadata.
        """

        metadata: dict[str, str] = {
            PROJECT_SLUG: project.project_slug,
            PROJECT_DISPLAY_NAME: project.project_display_name,
            DOCUMENT_TYPE: record.document_type,
            RESOLVED_COMMIT_SHA: project.resolved_commit_sha,
            CORPUS_FINGERPRINT: corpus_fingerprint,
            GENERATION_FINGERPRINT: record.generation_fingerprint,
            FILE_NAME: record.filename,
        }
        if record.component_id is not None:
            metadata[COMPONENT_ID] = record.component_id
        if project.repository_url is not None:
            metadata[REPOSITORY_URL] = project.repository_url
        validate_portfolio_metadata(metadata, allow_header_path=False)
        return Document(text=document.text, id_=record.filename, metadata=metadata)
