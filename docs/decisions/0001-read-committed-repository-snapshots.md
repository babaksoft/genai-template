# ADR-0001: Read Committed Repository Snapshots Through a Provider Boundary

**Status:** Accepted

## Context

Portfolio corpus generation must be repeatable even when a local repository has
staged, modified, or untracked files. A later implementation must also support
snapshots obtained from GitHub without changing downstream generation logic.

## Decision

Define a repository snapshot-reader boundary with one canonical snapshot result.
Implement `local_git` first: resolve a ref to a commit and read tracked blobs from
the local Git object database without checking out or modifying the repository.
Working-tree changes are excluded. A future GitHub reader will resolve its input to
a commit SHA and return the same snapshot model.

Repository contents are data only; snapshot readers never execute repository code.

## Consequences

- A snapshot is tied to an immutable commit and is independent of working-tree state.
- Local operation requires Git and access to the repository object database.
- Dirty-working-tree support, if added, is a separate explicitly non-reproducible
  mode.
- GitHub support can be added behind the reader boundary without changing corpus
  planning or generation.
