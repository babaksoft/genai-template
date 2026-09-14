# Source Citation Implementation Plan

## Summary

The complete feature is moderately cross-cutting, but not large enough to justify many incremental contracts. Implement it in two mergeable slices:

1. Backend and API citation support.
2. Streamlit presentation and trace visibility.

Backward compatibility is explicitly out of scope. New response fields are required, `ContextBuilder.build()` can change its return type directly, and no legacy response parsing or schema aliases are needed.

## Slice 1 — Backend and API Citations

Deliver a complete citation-capable `/answer` API that can be tested independently of Streamlit.

- Add `CitationSource`, `CitationWarning`, and `CitationContext` schemas.
  - A source contains `label`, `chunk_id`, safe `document_name`, optional `section`, exact chunk `content`, retrieval `distance`, and `cited`.
  - A warning contains `code`, `message`, and offending `labels`.
  - `CitationContext` contains the formatted context text and ordered sources.
- Change `ContextBuilder.build()` to return `CitationContext`.
  - Label chunks `S1`, `S2`, etc. in retrieval order.
  - Keep separate labels for multiple chunks from the same document.
  - Use `file_name`, falling back to `document_id`, and reduce it to a basename.
  - Use `header_path` as the optional section.
  - Never expose or interpolate absolute `file_path` values.
- Format model context as labeled source blocks containing document name, optional section, and chunk content.
- Update the prompt to require inline `[S<n>]` citations after supported claims, prohibit labels outside the context, avoid a generated bibliography, and allow uncited “I don’t know” responses.
- Add a citation resolver that:
  - Recognizes only `\[S[1-9][0-9]*\]`.
  - Marks supplied sources as cited.
  - Collects unique unsupported labels in numeric order.
  - Returns one `unsupported_citation_labels` warning when necessary.
  - Leaves the generated answer unchanged.
- Update `RagService` to build labeled context, generate the answer, resolve citations, and return sources and warnings.
- Replace `RagResult.retrieved_chunks` with `sources` and `citation_warnings`; retrieval chunks remain local to the service for metrics.
- Extend `AnswerResponse` with required `sources` and `citation_warnings` fields and return them from `POST /api/v1/answer`.
- Do not change request fields, status codes, vector-store payloads, index fingerprints, or database models.

Tests for this slice:

- Context construction covers ordering, duplicate documents, Markdown sections, missing metadata, safe basenames, and empty retrieval.
- Citation resolution covers valid, repeated, out-of-order, absent, and unsupported labels, plus ignored forms such as `[S0]`, `[S-1]`, and `[source]`.
- Service tests verify labeled context reaches the prompt, sources receive correct `cited` flags, warnings do not fail the run, and metrics persist normally.
- API tests assert the exact extended response and preserve existing 404, 409, and 422 behavior.
- Existing context, prompt, service, integration, and schema tests are updated directly for the new contracts; no compatibility tests are retained.

## Slice 2 — UI, Observability, and Documentation

Consume the completed API contract without changing citation semantics.

- Update API-client tests and fixtures to require and deserialize `sources` and `citation_warnings`.
- Render generated answers as Markdown so `[S1]` labels remain readable.
- Add a Sources section below the answer:
  - Display every context source in label order.
  - Mark cited and uncited sources.
  - Show document name, optional section, retrieval distance, and exact content in expanders.
  - Show an informative empty state when retrieval supplied no sources.
- Render each citation warning with `st.warning`; do not suppress or rewrite unsupported labels in the answer.
- Extend the existing `rag.answer` span with:
  - Total source count.
  - Number of cited sources.
  - Unsupported-label count.
  - Unsupported labels serialized as metadata when present.
- Keep source contents out of new citation-specific trace attributes; existing prompt/context tracing behavior remains unchanged.
- Update the README answer example and feature description to document inline labels, structured sources, warning behavior, and response-only retention.
- Run `./scripts/check.sh` after each slice. Run integration tests only when their external services are available, and avoid assertions requiring nondeterministic LLM citation wording.

## Final Public Contract

```json
{
  "answer": "FastAPI is an ASGI web framework [S1].",
  "metrics": {},
  "sources": [
    {
      "label": "S1",
      "chunk_id": "guide.md-001",
      "document_name": "guide.md",
      "section": "/Introduction/",
      "content": "Exact retrieved chunk text...",
      "distance": 0.12,
      "cited": true
    }
  ],
  "citation_warnings": [
    {
      "code": "unsupported_citation_labels",
      "message": "The answer references labels not present in its context.",
      "labels": ["S9"]
    }
  ]
}
```

## Assumptions

- Citation labels are request-local; chunk IDs remain the underlying evidence identifiers.
- Every retrieved context chunk is returned, including uncited chunks.
- Invalid labels generate warnings but never trigger retries or request failure.
- Validation confirms that labels resolve to supplied evidence; semantic entailment is outside this version.
- Answers, citations, and retrieved content are not persisted, so no migration is required.
- Since the project has not launched, clean final interfaces take priority over preserving internal or wire compatibility.
