# Portfolio RAG Stage 2 — Flat Ingestion, Provenance, and Index Freshness

**Status:** Proposed

## Goal

Index a validated, manifest-backed Portfolio corpus without introducing a second
directory-ingestion architecture, preserve project and generation provenance on
every chunk and citation, and refuse to answer from an index that does not represent
the currently published corpus.

Stage 2 ends with explicit, observable index rebuilds. It does not automatically
rebuild after publication, retain historical vector collections, add evaluation
trials, filter retrieval by project, or add conversations.

## Baseline and Contract Gap

Stages 0 and 1 already provide immutable repository snapshots, deterministic
project-prefixed Markdown filenames, canonical manifests, stable corpus
fingerprints, and atomic publication through an immutable release symlink.

The current RAG path still:

- loads Markdown without interpreting `manifest.json`;
- lets reader-generated absolute paths influence document and chunk identity;
- exposes only document basename and section in citations;
- treats collection existence as proof that an index is usable; and
- deletes and replaces a collection without persisting build intent or outcome.

There is also one deliberate Stage 1 limitation to resolve at the consumer
boundary. Its manifest describes one project, while the Stage 2 exit criteria
require documents from several projects to coexist in one flat corpus. Stage 2
therefore introduces a versioned, multi-project manifest projection while retaining
read compatibility with the Stage 1 single-project manifest. The Stage 1 generation
command may continue to emit one project per run; multi-project generation
orchestration remains out of scope.

## Decisions Fixed for This Stage

- Keep `TextReader` as the generic flat Markdown/text reader. Add a manifest-aware
  Portfolio loader around it instead of teaching generic ingestion about Portfolio
  workflow domain types.
- Resolve `data/portfolio` to its immutable release directory once at the start of
  a read. Manifest validation and document loading use that pinned path so an atomic
  publication switch cannot mix two releases in one build.
- Introduce manifest schema v2 with an ordered project registry and an explicit
  `project_slug` on every document record. Normalize both the existing v1 shape and
  v2 into one consumer model. New Stage 1 publications use v2; already published v1
  releases remain readable and keep their original fingerprints.
- A v2 manifest may describe several projects, but every Markdown filename remains
  globally unique and project-prefixed. Stage 2 supplies and tests the manifest
  construction/validation boundary; a CLI that composes several project releases is
  deferred unless implementation proves it necessary for the first experiment.
- Validate a published corpus without reopening repositories: canonical manifest
  bytes, safe flat file set, document byte sizes and hashes, corpus fingerprint,
  project references, and filename ownership are sufficient at ingestion time.
  Snapshot evidence membership remains a producer-side Stage 1 invariant.
- Use the project-prefixed filename as the portable document ID. Use a documented,
  deterministic filename-plus-ordinal format for chunk IDs. Absolute paths,
  publication release paths, timestamps, and vector-store-specific IDs never enter
  either identity.
- Define stable metadata keys rather than passing through arbitrary reader metadata.
  Portfolio chunks carry project slug/display name, document type, optional
  component ID, repository URL when configured, resolved commit, corpus
  fingerprint, and document generation fingerprint. `file_name` and Markdown
  `header_path` remain available for current splitters and citations.
- Preserve non-manifest sources. They continue through generic ingestion and use the
  current collection-existence behavior; manifest freshness is reported as
  `untracked` rather than fabricating Portfolio provenance.
- Keep collection names derived from source ID and index-affecting RAG configuration.
  A corpus change makes that collection stale; it does not select another collection
  and does not delete any historical collection automatically.
- Persist every rebuild attempt before the existing destructive collection reset.
  The newest attempt for a source/index pair must have succeeded before the
  collection can be used. This prevents an older successful record from blessing a
  collection left absent or partial by a later failed rebuild.
- Determine manifest freshness from the latest successful build, but determine
  operational availability from both that successful build and the newest attempt.
  A building or failed newer attempt makes the index unavailable even when its
  corpus fingerprint happens to match.
- A successful status additionally requires the collection to exist and its current
  count to equal the persisted successful chunk count. The SQL record is not treated
  as proof that external vector-store state still exists.
- Rebuild failures store a bounded, sanitized operator-facing summary. Source text,
  prompts, generated prose, embeddings, credentials, and full provider payloads are
  never persisted in build records or attached to custom trace attributes.
- Stale, building, failed, missing, or inconsistent indexes fail before a RAG `Run`
  is created or an LLM is invoked. The API reports an actionable conflict and the UI
  requires an explicit rebuild.

## Proposed Package Changes

```text
src/genai_template/
  components/readers/
    portfolio_reader.py
  db/models/
    index_build.py
  schemas/
    corpus.py
    index_build.py
  services/
    index_build_service.py
  workflow/portfolio/
    corpus/
      loading.py
    domain/
      manifest.py              # v1 input plus v2/normalized contracts

alembic/versions/
  <revision>_add_index_builds.py

tests/
  components/readers/test_portfolio_reader.py
  db/test_index_build_migration.py
  services/test_index_build_service.py
  workflow/portfolio/corpus/test_loading.py
```

Names may be consolidated when a module would otherwise be trivial. Manifest
parsing, reader enrichment, splitters, persistence, freshness policy, and API/UI
presentation must remain independently testable.

## Slice 1 — Versioned Corpus Consumption Contract

**Status: Proposed**

### Outcome

A caller can pin and validate one published Portfolio release, normalize either the
Stage 1 manifest or a multi-project v2 manifest, and obtain stable corpus provenance
without loading a vector store or database.

### Implementation

- Retain the existing manifest v1 model as an immutable input contract.
- Add manifest v2 models containing:
  - an ordered, non-empty project registry with slug, display name, repository URL,
    requested ref, resolved commit, and source fingerprint;
  - generation profile/provider/model and stable configuration provenance per
    project, so projects generated differently are representable;
  - document records with an explicit owning project slug plus the existing type,
    component, evidence, artifact, generation, content-hash, and size fields; and
  - one corpus fingerprint covering the complete ordered project and document
    projection plus Markdown byte hashes.
- Make the Stage 1 builder publish v2 for a single project without changing its
  generation, rendering, cache, or atomic-publication behavior. Do not rewrite or
  reinterpret existing v1 release bytes.
- Add a normalized immutable consumer model used by ingestion. Convert v1 by
  projecting its top-level project/generation values into one project entry and by
  assigning that project to every document.
- Add consumer-side corpus loading that:
  - validates the source path and resolves a publication symlink exactly once;
  - rejects a release path outside the configured source only when source
    registration invariants require that restriction;
  - reads strict UTF-8 canonical JSON with explicit schema dispatch;
  - checks safe flat files, unique globally project-prefixed filenames, known project
    references, content sizes/hashes, and the schema-appropriate corpus fingerprint;
  - excludes `manifest.json` from documents; and
  - returns the pinned release path, normalized manifest, and ordered document
    records without exposing machine-local paths as stable metadata.
- Separate consumer validation from Stage 1 snapshot validation. The latter still
  proves evidence membership when producing a release.
- Fail closed on unknown schema versions, non-canonical JSON, broken symlinks,
  nested/symlinked document entries, filename collisions, orphan project references,
  changed bytes, or fingerprint disagreement.

### Verification

- Test canonical one-project v2 construction and fingerprint stability.
- Test normalization of an unchanged v1 fixture without changing its stored corpus
  fingerprint.
- Test a synthetic two-project v2 corpus with different repositories, commits,
  models, and component IDs.
- Test publication-pointer switching during a read and prove all bytes come from the
  initially pinned release.
- Test every malformed-manifest and malformed-directory failure before any reader,
  database, embedder, or vector-store operation.
- Re-run Stage 1 manifest, corpus validation, publisher, workflow, and CLI tests.

Run:

```bash
uv run pytest tests/workflow/portfolio/corpus \
  tests/workflow/portfolio/workflow \
  tests/workflow/portfolio/cli -v
```

### Acceptance

- Existing v1 releases remain consumable and byte-identical.
- New one- or multi-project v2 releases have one unambiguous normalized provenance
  model.
- One read cannot combine the manifest or Markdown bytes of different atomic
  releases.
- No LLM, SQL, embedding, or vector-store operation occurs in this slice.

## Slice 2 — Manifest-Aware Reader, Metadata, and Stable Identities

**Status: Proposed**

### Outcome

The existing indexing pipeline can read a valid Portfolio corpus, and every chunk
reaches either vector backend with portable identity and complete provenance.

### Implementation

- Define a typed `LoadedCorpus`/`LoadedDocuments` result carrying ordered
  `Document` objects and optional normalized corpus provenance. Introduce the
  smallest reader protocol needed by `IndexingPipeline` so both generic and
  Portfolio readers implement the same boundary.
- Add a Portfolio reader that layers on `TextReader` after Slice 1 validation. Map
  documents to manifest records by exact safe basename and reject missing,
  duplicated, extra, or reader-renamed documents.
- Replace reader-supplied identity and path metadata for Portfolio documents with a
  controlled metadata projection. In particular, remove absolute `file_path` and
  release-directory values before splitting.
- Set the canonical Portfolio document ID to its project-prefixed filename and load
  files in manifest order.
- Centralize Portfolio metadata key names and validation. Required values on every
  Portfolio document and chunk are:
  - `project_slug` and `project_display_name`;
  - `document_type` and optional `component_id`;
  - optional `repository_url` and required `resolved_commit_sha`;
  - `corpus_fingerprint` and `generation_fingerprint`; and
  - portable `file_name`, plus `header_path` when produced by the Markdown splitter.
- Update sentence and Markdown splitters to derive document identity from the
  canonical document ID, preserve controlled metadata, and emit deterministic chunk
  IDs from filename and zero-padded ordinal. Reject duplicate document or chunk IDs
  before embedding.
- Extend the indexing result with the loaded corpus fingerprint when present, while
  keeping existing document/chunk counts and timing.
- Verify Chroma and Qdrant round-trip every supported metadata value. Backend-native
  Qdrant UUID conversion remains an implementation detail; retrieved chunks expose
  the canonical chunk ID.
- Keep generic `.md`/`.txt` ingestion backward compatible. Generic sources do not
  receive invented Portfolio fields or manifest freshness guarantees.

### Verification

- Test exact document metadata for overview, architecture, testing/operations, and
  component records.
- Run both splitters against the same two-project fixture and assert unique,
  deterministic document/chunk IDs with full provenance on every chunk.
- Test that changing the checkout path, release directory, or publication symlink
  does not alter IDs or metadata.
- Test rejection of arbitrary reader paths, record mismatches, duplicate IDs, and
  metadata loss.
- Add Chroma and Qdrant persistence/search round-trip tests for all provenance keys.
- Re-run the existing generic reader, splitter, pipeline, and vector-store tests.

Run:

```bash
uv run pytest tests/components/readers \
  tests/components/splitters \
  tests/pipelines/test_indexing_pipeline.py \
  tests/stores/vector -v
```

### Acceptance

- Every Portfolio chunk contains the exact project, repository revision, corpus,
  document, component, and generation provenance represented by its manifest.
- Two projects with similarly named logical documents cannot collide because the
  manifest enforces project-prefixed filenames and splitters use those names as the
  identity root.
- Neither chunk identity nor indexed metadata contains an absolute local path.
- Legacy flat sources still index through the generic reader.

## Slice 3 — Provenance-Rich Context and Citations

**Status: Proposed**

### Outcome

Retrieved Portfolio chunks give the model and API caller enough information to
distinguish projects and audit every citation back to the generated document and
repository revision.

### Implementation

- Extend `CitationSource` with optional, backward-compatible provenance fields for
  project slug/display name, document type, component ID, repository URL, resolved
  commit, corpus fingerprint, and generation fingerprint.
- Build those fields only from validated chunk metadata. Portfolio metadata that is
  partially present or malformed is an indexing/retrieval contract error rather
  than a silently incomplete Portfolio citation; generic sources may omit the whole
  Portfolio provenance group.
- Include concise project, repository, revision, document type/component, and
  Markdown section labels in the model context. Omit absent optional repository URLs
  rather than rendering `None`.
- Continue using request-local `[S<n>]` labels and the current unsupported-label
  resolver. Citation resolution must preserve all new provenance fields unchanged
  when setting `cited`.
- Keep the safe basename as the displayed generated document name. Never expose the
  registered source directory or immutable release path.
- Add provenance counts/identities to spans only where they are bounded and useful;
  do not attach source content beyond the existing instrumentation policy.

### Verification

- Test context formatting for two projects with the same section name and different
  commits.
- Test component and project-level documents, missing optional repository URLs, and
  generic-source compatibility.
- Test that malformed partial Portfolio metadata fails before LLM generation.
- Test citation resolution and API serialization preserve every provenance value.
- Test that absolute reader/release paths never appear in context, response models,
  logs, or new custom span attributes.

Run:

```bash
uv run pytest tests/components/context \
  tests/schemas/test_citation.py \
  tests/api/test_answer.py -v
```

### Acceptance

- A cited Portfolio source identifies its generated filename, project, repository
  revision, corpus, and generation artifact.
- Cross-project context is visibly disambiguated for both the LLM and API caller.
- Existing citation labels and generic-source responses remain compatible.

## Slice 4 — Persistent Index-Build Attempts and Safe Lifecycle

**Status: Proposed**

### Outcome

Every explicit rebuild has a durable identity and terminal outcome, including
failures that may have invalidated the selected collection.

### Implementation

- Add an Alembic revision and `IndexBuild` persistence model with documented fields:
  - primary key, source ID, and the RAG config ID that requested the build;
  - deterministic collection name and full index-configuration fingerprint;
  - nullable corpus fingerprint for legacy non-manifest sources;
  - status (`building`, `succeeded`, or `failed`);
  - started/finished timestamps and measured indexing duration;
  - nullable document/chunk counts until success; and
  - nullable bounded failure code/detail for operator diagnosis.
- Add indexes supporting newest-attempt and newest-successful lookups by source,
  collection, and index fingerprint. Do not make corpus fingerprint unique: explicit
  unchanged rebuilds are valid audit events.
- Add `Source.index_builds` and `RagConfig.index_builds` relationships with explicit
  deletion behavior. A source cascade removes its local audit rows; referenced RAG
  configurations remain restricted.
- Implement a focused index-build persistence service with start, succeed, fail,
  latest-attempt, and latest-successful operations. Each transition uses a short SQL
  transaction and rejects invalid terminal-state rewrites.
- In `SourceService.rebuild_index`:
  1. acquire the existing per-collection process lock without waiting, returning a
     focused in-progress conflict when another request in this process owns it;
  2. validate source/config and pin/load the corpus;
  3. persist and commit a `building` attempt before deleting the collection;
  4. delete and rebuild the selected collection;
  5. verify store count equals the pipeline's chunk count;
  6. mark success with counts, timing, and the pinned corpus fingerprint; or
  7. mark failure with sanitized bounded details and re-raise the original focused
     error.
- Always release the process lock in `finally`. A persisted `building` row left by a
  dead process does not itself prevent a later explicit rebuild: the newer attempt
  supersedes it for availability decisions.
- If the process dies after step 3, the durable `building` row intentionally keeps
  the index unavailable until an operator explicitly rebuilds. Automatic recovery,
  leases, and distributed locks are deferred.
- Add build IDs, status, fingerprints, counts, and timing to observability. Never
  record document contents, embeddings, secrets, or raw provider payloads.

### Verification

- Test migration upgrade/downgrade, foreign keys, indexes, nullability, status
  constraints, and model metadata/doc descriptions.
- Test valid transitions, double completion, failure sanitization/truncation, and
  newest-attempt/newest-successful ordering.
- Inject failures before delete, during delete/load/split/embed/upsert, during count
  verification, and while marking success. Assert the durable outcome is honest at
  every point where a build attempt exists.
- Test that a rebuild which indexed a pinned old release succeeds with that release's
  fingerprint even if publication switches during the build; freshness is evaluated
  separately in Slice 5.
- Re-run source service, migration, pipeline, Chroma, and Qdrant tests.

Run:

```bash
uv run pytest tests/db \
  tests/services/test_index_build_service.py \
  tests/services/test_source_service.py \
  tests/pipelines \
  tests/stores/vector -v
```

### Acceptance

- A build attempt is durable before the selected collection can be destroyed.
- Success records exactly the pinned corpus and verified stored chunk count.
- Any rebuild failure after attempt creation leaves a terminal failed row or, on
  process death, a conservative building row; neither can be mistaken for a usable
  current index.
- Unchanged explicit rebuilds create audit records but retain the same deterministic
  collection name.

## Slice 5 — Freshness Policy and Answer Gate

**Status: Proposed**

### Outcome

One domain service reports an actionable index state, and Portfolio answers cannot
run against an unbuilt, stale, rebuilding, failed, missing, or inconsistent index.

### Implementation

- Define a typed `IndexStatus` projection with source/config/collection/index
  identity, current corpus fingerprint, built corpus fingerprint, latest build ID
  and status, counts/timestamps, an availability boolean, and a machine-readable
  reason.
- Use explicit reason values such as `current`, `unbuilt`, `stale`, `building`,
  `failed`, `collection_missing`, `count_mismatch`, `backend_unavailable`,
  `corpus_invalid`, and `untracked`. Keep status derivation in the service layer
  rather than duplicating it in routes, RAG execution, or Streamlit.
- For a manifest-backed source:
  1. resolve and validate the currently published manifest;
  2. find the latest successful build for the selected source/index fingerprint;
  3. compare its corpus fingerprint with the current manifest;
  4. ensure no newer building or failed attempt invalidates the collection; and
  5. confirm collection existence and exact count.
- Treat a changed corpus fingerprint as `stale` even when filenames, collection
  name, and chunk count are unchanged.
- For generic sources, report `untracked` and retain collection existence as the
  availability rule. This is a compatibility path, not a claim of manifest
  freshness.
- Replace the RAG service's direct `store.exists()` check with this status policy.
  Raise one focused `IndexUnavailableError` containing the reason and rebuild action;
  retain `IndexNotBuiltError` as an alias or compatibility subclass if needed.
- Perform the gate before starting a `Run`, retrieval, or LLM generation. Attach the
  successful build ID and corpus fingerprint to answer spans for auditability.
- Document the small race boundary between status inspection and retrieval. Cross-
  process read/write exclusion or backend aliases are deferred; persisted
  `building` state and current service ordering provide fail-closed behavior for the
  supported process model.

### Verification

- Table-test every status and precedence combination, especially:
  - old success plus newly changed manifest;
  - matching old success plus newer failed/building attempt;
  - matching success plus absent collection or count mismatch;
  - a build pinned to the prior release after publication switched;
  - retrieval/LLM factories not called for every unavailable state; and
  - generic source compatibility.
- Assert unavailable answers do not create a `Run` row.
- Assert current answers record build/corpus identity in bounded spans without source
  content.
- Re-run all RAG and experiment service tests.

Run:

```bash
uv run pytest tests/services/test_source_service.py \
  tests/services/test_rag_service.py \
  tests/services/test_rag_observability.py \
  tests/integration/test_registry_workflow.py -v
```

### Acceptance

- Publishing a different corpus makes the existing Portfolio index unavailable
  immediately, without renaming or deleting its collection.
- Only the newest valid successful rebuild of the currently published corpus can be
  used for answers.
- No unavailable-index request reaches retrieval, starts a durable RAG run, or calls
  an LLM.
- Legacy sources retain their pre-Stage 2 behavior and are clearly labeled
  `untracked`.

## Slice 6 — Index API, Streamlit Workflow, and Stage Integration

**Status: Proposed**

### Outcome

Operators can see why a selected index is unavailable, explicitly rebuild it, and
inspect project/repository provenance on answer citations through the existing API
and Streamlit workflow.

### Implementation

- Add `GET /sources/{source_id}/indexes/{rag_config_id}` returning the typed current
  `IndexStatus` projection.
- Change the existing `PUT` endpoint to return the persisted successful build plus
  the resulting current status. Keep document count, chunk count, and duration fields
  available to current clients.
- Map unknown source/config to `404`, invalid corpus to a focused validation error,
  concurrent/already-running rebuild to `409`, and unavailable answer states to
  `409` with a machine-readable reason and explicit rebuild endpoint/action.
- Extend the API client with index-status lookup and the expanded rebuild response.
- In Streamlit, once an experiment and configuration are selected:
  - load and display current/indexed corpus fingerprints in abbreviated form;
  - distinguish current, stale, building, failed, missing, and untracked states;
  - disable Ask for unavailable manifest-backed states;
  - keep rebuild explicit and show persisted build ID, counts, duration, and any
    safe failure guidance; and
  - refresh status after a rebuild.
- Extend answer source rendering with project display name/slug, generated document
  type/component, repository URL when present, and full resolved commit. Repository
  deep links remain Stage 6 master-plan scope.
- Update README/operator documentation for generation → source registration → status
  → explicit rebuild → answer, including stale-index behavior and legacy-source
  semantics.
- Add one end-to-end test using a temporary published Portfolio release, SQLite, a
  deterministic fake embedder/store/LLM, and the API boundary:
  1. register source/config/experiment;
  2. observe `unbuilt`;
  3. rebuild and observe `current`;
  4. answer and inspect citation provenance;
  5. atomically publish a changed release and observe `stale` plus rejected answer;
  6. rebuild and answer from the new corpus; and
  7. inject a failed rebuild and verify answers remain unavailable.

### Verification

- Test API response schemas, status codes, error reason payloads, and compatibility
  fields on rebuild responses.
- Test API client requests and Streamlit rendering/disablement for every status.
- Test citation presentation with and without repository URLs.
- Run Stage 0–2 focused tests, the end-to-end lifecycle test, then the complete
  non-integration quality suite with Phoenix disabled.

Run:

```bash
uv run pytest tests/workflow/portfolio \
  tests/components/readers \
  tests/components/splitters \
  tests/components/context \
  tests/db \
  tests/pipelines \
  tests/services \
  tests/api \
  tests/ui \
  tests/stores/vector -v
PHOENIX_ENABLED=false ./scripts/check.sh
```

### Acceptance

- API and UI expose one consistent, actionable freshness decision.
- The UI never presents Ask as available for a stale or operationally invalid
  Portfolio index.
- A successful rebuild serves the current corpus fingerprint and provenance-rich
  citations.
- The full non-integration quality suite passes without Phoenix, Ollama, Qdrant
  server, or an external LLM.

## Stage 2 Completion Criteria

Stage 2 is complete only when all six slices are implemented and verified, and:

- every Portfolio chunk and citation carries correct document, project, repository
  revision, corpus, and generation provenance;
- one flat v2 corpus can represent and index several projects without document or
  chunk-ID collisions;
- existing Stage 1 v1 releases and generic flat corpora retain documented
  compatibility behavior;
- changing only the published corpus fingerprint makes the selected Portfolio index
  unavailable until an explicit successful rebuild;
- the newest destructive rebuild attempt cannot be bypassed by an older success;
- SQL build records, vector-store existence/counts, and current manifest identity
  must agree before an answer runs;
- cited answers identify the generated document, project, and repository revision;
- source text, generated prose, embeddings, credentials, and machine-local release
  paths are absent from build records and new custom trace attributes;
- all new functions, classes, and methods have complete type hints and Google-style
  docstrings, and all new persistence fields include concise `doc` descriptions; and
- `PHOENIX_ENABLED=false ./scripts/check.sh` passes.

## Explicitly Deferred

- automatic rebuilds triggered by corpus publication;
- scheduled refreshes, leases, distributed rebuild locks, and crash recovery;
- atomic vector-collection aliases or blue/green collection swaps;
- historical vector collections, automatic deletion, and retention policies;
- a CLI or scheduler that generates or composes several projects in one run;
- project metadata filtering, reranking, hybrid retrieval, and other Stage 4 work;
- evaluation datasets, durable evaluation trials, and configuration comparison;
- conversation persistence and history-aware retrieval;
- GitHub repository links beyond displaying available repository/revision metadata;
- semantic validation that an evidence path entails a generated claim; and
- migration of legacy generic corpora into Portfolio manifests.

## Commit and Review Guidance

- Keep one reviewable commit per slice.
- Each slice must leave its focused tests passing and preserve generic corpus
  ingestion.
- Commit the Alembic revision with the `IndexBuild` model in Slice 4; do not rewrite
  the existing initial revision.
- Do not commit generated caches, immutable corpus releases, the `data/portfolio`
  symlink, vector-store data, or local SQLite contents.
- Run the complete check script only after Slice 6, while running the focused command
  listed under every earlier slice before review.
