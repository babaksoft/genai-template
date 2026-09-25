# Portfolio RAG Stage 0 — Snapshot and Corpus Contract

**Status:** Implemented

## Goal

Implement the read-only foundation that converts one local Git repository at an
immutable commit into a validated, normalized, content-addressed
`RepositorySnapshot` and a logical summary plan.

Stage 0 does not invoke an LLM, generate Markdown corpus documents, write to
`data/portfolio`, access GitHub, update SQL, or touch a vector store. Its final
integration point is a read-only inspection command that makes the exact inputs for
Stage 1 visible and verifiable.

## Decisions Fixed for This Stage

- The supported source mode is `local_git`.
- A ref defaults to `HEAD` and is resolved to a full commit SHA.
- Files are read from committed Git objects, not from working-tree paths.
- Modified, staged, and untracked files are ignored.
- Snapshot reading never checks out a ref or changes the inspected repository.
- File selection is allowlist-first, with explicit excludes applied afterward.
- Only UTF-8 text files within a configured size limit are accepted.
- Repository-relative paths use normalized POSIX separators.
- Summary units are explicit logical groups rather than every discovered Python
  package.
- Absolute repository paths, filesystem timestamps, and Git command output ordering
  do not influence fingerprints.
- The configuration shape permits multiple project entries, but the first example
  and end-to-end acceptance path use one project.

## Implemented Package Layout

```text
src/genai_template/workflow/portfolio/
  __init__.py
  inspect_snapshot.py              # compatibility CLI entry point
  adapters/repositories/local_git.py
  cli/inspect.py
  config/
    __init__.py
    loader.py
    models.py
    profiles/local.yml
  domain/
    errors.py
    snapshot.py
    summary.py
  inspection/
    reports.py
    service.py
  ports/repository.py
  snapshot/
    planning.py
    selection.py

tests/workflow/portfolio/
  adapters/repositories/test_local_git_snapshot_reading.py
  config/test_config_loading.py
  domain/test_domain_models.py
  inspection/test_snapshot_inspection.py
  snapshot/
    test_snapshot_selection.py
    test_summary_planning.py
```

The Stage 0 migration preserved the original curated package exports and legacy CLI
module while separating domain, port, adapter, application-service, and interface
responsibilities.

## Public Model and Protocol Direction

The Stage 0 types should be immutable Pydantic models with forbidden extra fields
where they cross configuration or workflow boundaries.

Conceptual configuration:

```yaml
version: 1

projects:
  - slug: genai-template
    display_name: GenAI Template
    repository:
      type: local_git
      path: .
      ref: HEAD
    selection:
      include:
        - README.md
        - pyproject.toml
        - src/**/*.py
        - tests/**/*.py
      exclude:
        - "**/__pycache__/**"
      max_file_bytes: 262144
    summary_units:
      - id: components-splitters
        paths:
          - src/genai_template/components/splitters/**/*.py
          - tests/components/splitters/**/*.py
```

Conceptual runtime types:

```text
RepositorySnapshot
- project_slug
- repository_url
- requested_ref
- resolved_commit_sha
- source_fingerprint
- files: tuple[SnapshotFile, ...]

SnapshotFile
- path
- text
- content_hash
- byte_size

SummaryPlan
- project_slug
- resolved_commit_sha
- source_fingerprint
- units: tuple[SummaryUnitPlan, ...]

SummaryUnitPlan
- unit_id
- input_fingerprint
- files: tuple[SnapshotFile, ...]
```

The reader protocol accepts a validated local-Git source specification and returns
commit identity plus tracked blob entries. Selection, decoding, normalization, and
summary planning remain separate services so a future GitHub reader can reuse them.

## Slice 1 — Typed Configuration and Domain Models

### Outcome

A caller can load and validate a Portfolio corpus configuration and construct the
canonical Stage 0 domain types without invoking Git.

### Implementation

- Add immutable typed models for:
  - portfolio configuration version;
  - project identity;
  - discriminated local-Git repository settings;
  - include/exclude selection settings and maximum file size;
  - logical summary-unit identifiers and path patterns;
  - committed repository entries;
  - normalized snapshot files;
  - repository snapshots; and
  - summary plans and planned units.
- Add a safe YAML loader that:
  - requires a mapping root and supported schema version;
  - rejects unknown keys and unsupported source types;
  - resolves relative local repository paths from `settings.REPO_ROOT`;
  - requires non-empty, normalized project slugs and refs;
  - rejects duplicate project slugs and duplicate unit IDs within a project;
  - rejects absolute repository-file patterns and patterns containing `..`; and
  - requires at least one include pattern and one summary unit.
- Add a documented one-project example configuration under
  `src/genai_template/workflow/portfolio/config/profiles/` using the current
  repository as a fixture target.
- Export only the intended public models and loader from the portfolio workflow
  package.

### Verification

- Test successful loading and immutable model behavior.
- Test relative path resolution.
- Test malformed YAML, unsupported versions and source types, unknown keys, duplicate
  slugs/unit IDs, empty lists, invalid slugs, unsafe patterns, and invalid size
  limits.
- Confirm this slice performs no Git subprocess calls.

Run:

```bash
uv run pytest tests/workflow/portfolio/config/test_config_loading.py -v
```

### Acceptance

- A valid one-project YAML file produces a fully validated configuration.
- Invalid configuration fails before repository access.
- No runtime behavior outside the new Stage 0 package changes.

## Slice 2 — Read-Only Local Git Snapshot Reader

### Outcome

A caller can resolve a local repository ref and enumerate/read the committed tracked
blobs without observing or changing the working tree.

### Implementation

- Define the repository-reader protocol needed by Stage 1.
- Implement `LocalGitSnapshotReader` using non-shell Git subprocess arguments.
- Validate that the configured path exists and is a Git work tree.
- Resolve `<ref>^{commit}` to a full commit SHA and reject missing or non-commit
  references clearly.
- Enumerate tracked entries from the resolved commit using a NUL-safe Git format so
  spaces and other valid filename characters remain unambiguous.
- Read blob bytes directly from the resolved commit rather than opening working-tree
  files.
- Obtain and normalize the `origin` URL when available; permit an explicit configured
  repository URL or a documented null value when no origin exists.
- Do not use `shell=True`, `git checkout`, `git switch`, `git reset`, `git clean`, or
  commands that write repository state.
- Translate Git failures into a focused application exception without hiding useful
  ref/path context.

### Verification

- Create temporary Git repositories in tests with deterministic commits.
- Verify ref resolution for `HEAD`, a branch, a tag, and an explicit commit SHA.
- After committing a file, modify it, stage a different version, and add an untracked
  file; verify that the reader still returns only the committed bytes.
- Verify paths containing spaces are read correctly.
- Verify the current branch and working-tree status are unchanged after reading.
- Test missing paths, non-repositories, missing refs, and absent origins.

Run:

```bash
uv run pytest tests/workflow/portfolio/test_local_git_reader.py -v
```

### Acceptance

- The reader returns the same commit and blob data for the same ref regardless of
  working-tree changes.
- Reading has no repository side effects and makes no network calls.
- The protocol does not expose local-Git implementation details to downstream
  selection and planning code.

## Slice 3 — Deterministic Selection, Normalization, and Planning

### Outcome

Committed blob entries become a stable `RepositorySnapshot`, and configured logical
units become a stable `SummaryPlan` suitable for Stage 1 LLM calls.

### Implementation

- Apply include patterns first and exclude patterns second against normalized
  repository-relative POSIX paths.
- Define and test pattern semantics once; do not depend on platform-specific path or
  glob ordering.
- Reject path traversal, directories, symlinks requiring filesystem traversal,
  submodule entries, Git LFS pointers, binary content, invalid UTF-8, and files over
  the configured byte limit with explicit errors or typed warnings as documented by
  the implementation.
- Normalize accepted text to UTF-8 strings with LF line endings and a documented
  terminal-newline rule.
- Sort files lexically by normalized path before hashing or planning.
- Calculate each file's SHA-256 content hash.
- Calculate `source_fingerprint` from stable selection settings plus selected paths
  and normalized content hashes. Exclude the absolute local path, timestamps, and
  command ordering.
- Resolve summary-unit path patterns against selected files.
- Reject empty units and accidental duplicate unit IDs; define whether deliberate
  file membership in more than one unit is allowed and cover that rule in tests.
- Calculate each unit input fingerprint from its identifier, stable configuration,
  selected paths, and content hashes. Prompt and model settings are added to the
  generation fingerprint in Stage 1, not this Stage 0 fingerprint.

### Verification

- Test include/exclude precedence and stable ordering.
- Test normalization of CRLF and missing terminal newlines.
- Test binary, invalid-encoding, oversized, traversal, symlink, submodule, and LFS
  cases.
- Test that identical selected contents at different absolute repository paths
  produce the same source and unit fingerprints.
- Test that selected content or selection-rule changes alter the relevant
  fingerprint.
- Test that working-tree changes and changes to excluded files do not alter the
  snapshot.
- Test unit membership, empty-unit rejection, and unit fingerprint isolation.

Run:

```bash
uv run pytest tests/workflow/portfolio/snapshot/test_snapshot_selection.py \
  tests/workflow/portfolio/snapshot/test_summary_planning.py -v
```

### Acceptance

- Snapshot and unit ordering and fingerprints are stable across repeated runs.
- Every planned unit contains only validated files from the canonical snapshot.
- A future snapshot reader can feed the same selection and planning services without
  changing their contracts.

## Slice 4 — Snapshot Inspection CLI and Stage Integration

### Outcome

A user can verify the exact committed inputs and planned LLM work for one repository
through a read-only command before Stage 1 is implemented.

### Implementation

- Add an inspection entry point under the portfolio workflow package.
- Accept a required configuration path and optional project-slug filter.
- Load configuration, resolve each selected local-Git snapshot, apply selection, and
  build the summary plan.
- Print a stable human-readable report containing:
  - project slug and display name;
  - repository URL when available;
  - requested ref and resolved commit SHA;
  - selected file count and total bytes;
  - source fingerprint;
  - each summary unit, its fingerprint, and its member paths; and
  - warnings for explicitly tolerated unsupported entries.
- Add an optional JSON report mode for automated inspection. Do not include source
  contents or absolute local paths in either report.
- Return non-zero status for configuration, repository, selection, or planning
  failures.
- Document the command, committed-snapshot semantics, and current exclusions.
- Add orchestration tests with Git and configuration boundaries exercised through
  temporary repositories; mock only where necessary for output/error isolation.

Proposed command:

```bash
uv run python -m genai_template.workflow.portfolio.cli.inspect \
  --config src/genai_template/workflow/portfolio/config/profiles/local.yml
```

### Verification

- Test deterministic text and JSON reports.
- Test project filtering, missing project selection, and propagated failures.
- Verify reports contain no file contents or absolute repository paths.
- Run all Stage 0 tests, then the complete non-integration quality suite.

Run:

```bash
uv run pytest tests/workflow/portfolio -v
./scripts/check.sh
```

### Acceptance

- The inspection command proves which immutable source material Stage 1 will use.
- Repeated inspection of the same commit and configuration produces the same stable
  report fields and fingerprints.
- Dirty working-tree state cannot alter the reported snapshot.
- No LLM, GitHub, corpus, database, vector-store, or external-service access occurs.
- `./scripts/check.sh` passes.

## Stage 0 Completion Criteria

Stage 0 is complete only when all four slices are implemented and verified, and:

- the public snapshot-reader boundary can support a future GitHub implementation;
- one local repository can be inspected at an exact commit without side effects;
- every selected file and summary unit has a stable content identity;
- configuration and error messages are sufficient to diagnose invalid repositories,
  refs, patterns, and source files;
- all new functions, classes, and methods have complete Google-style docstrings;
- no secrets or source contents appear in inspection reports or custom telemetry;
  and
- the master plan's Stage 0 exit criteria are satisfied.

## Deferred Until Later Stages

- LlamaIndex Workflow events and execution;
- LLM clients, prompts, structured summaries, caching, and cost accounting;
- corpus Markdown rendering, manifests, validation, and atomic publication;
- manifest-aware RAG ingestion and project metadata;
- index-build persistence and stale-index rejection;
- GitHub API/archive access and authentication;
- dirty-working-tree snapshots;
- scheduled execution and `--if-changed` corpus generation;
- multi-project corpus generation in one run; and
- automatic package discovery or agentic repository exploration.

## Commit and Review Guidance

- Keep one reviewable commit per slice.
- Each slice must leave its focused tests passing and must not depend on unfinished
  code from a later slice.
- Preserve unrelated user files and changes, including existing material under
  `work/ongoing/`.
- Run `./scripts/check.sh` after Slice 4 before declaring Stage 0 complete.
