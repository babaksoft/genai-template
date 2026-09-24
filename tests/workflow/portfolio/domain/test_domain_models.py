"""Tests for Portfolio Stage 0 domain-model behavior."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from genai_template.workflow.portfolio import (
    CommittedRepositoryEntry,
    RepositoryReadResult,
    RepositorySnapshot,
    SnapshotFile,
    SummaryPlan,
    SummaryUnitPlan,
)


def test_domain_models_are_immutable_and_composable() -> None:
    """Canonical Stage 0 domain values compose without repository access."""

    content_hash = "1" * 64
    commit_sha = "2" * 40
    source_fingerprint = "3" * 64
    unit_fingerprint = "4" * 64
    file = SnapshotFile(
        path="src/example.py",
        text="print('example')\n",
        content_hash=content_hash,
        byte_size=17,
    )
    snapshot = RepositorySnapshot(
        project_slug="sample-project",
        repository_url=None,
        requested_ref="HEAD",
        resolved_commit_sha=commit_sha,
        source_fingerprint=source_fingerprint,
        files=(file,),
    )
    plan = SummaryPlan(
        project_slug=snapshot.project_slug,
        resolved_commit_sha=snapshot.resolved_commit_sha,
        source_fingerprint=snapshot.source_fingerprint,
        units=(
            SummaryUnitPlan(
                unit_id="application-core",
                input_fingerprint=unit_fingerprint,
                files=snapshot.files,
            ),
        ),
    )
    entry = CommittedRepositoryEntry(
        path="src/example.py",
        mode="100644",
        object_type="blob",
        object_id=commit_sha,
        content=b"print('example')\n",
    )

    assert plan.units[0].files == snapshot.files
    assert entry.content == b"print('example')\n"
    with pytest.raises(ValidationError):
        file.text = "changed"


@pytest.mark.parametrize(
    "model_type",
    [
        CommittedRepositoryEntry,
        RepositoryReadResult,
        SnapshotFile,
        RepositorySnapshot,
        SummaryUnitPlan,
        SummaryPlan,
    ],
)
def test_public_models_document_classes_and_fields(
    model_type: type[BaseModel],
) -> None:
    """Every public Pydantic model documents its attributes and schema fields."""

    assert model_type.__doc__ is not None
    assert "Attributes:" in model_type.__doc__
    assert all(field.description for field in model_type.model_fields.values())
