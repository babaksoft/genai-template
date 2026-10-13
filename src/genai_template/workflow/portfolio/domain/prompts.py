from __future__ import annotations

import hashlib

from pydantic import Field, computed_field

from genai_template.workflow.portfolio.domain.base import ImmutableDomainModel
from genai_template.workflow.portfolio.domain.summaries import (
    ArtifactKind,
    SUMMARY_SPECIFICATIONS,
)
from genai_template.workflow.portfolio.prompts.prompts_v1 import (
    ARCHITECTURE_PROMPT,
    ARCHITECTURE_PROMPT_ID,
    COMPONENT_PROMPT,
    COMPONENT_PROMPT_ID,
    OVERVIEW_PROMPT,
    OVERVIEW_PROMPT_ID,
    PROMPT_TEMPLATE,
    SAFETY_PROMPT,
    TESTING_OPERATIONS_PROMPT,
    TESTING_OPERATIONS_PROMPT_ID,
)


class PromptDefinition(ImmutableDomainModel):
    """A stable versioned prompt template.

    Attributes:
        prompt_id:
            Stable identifier including the prompt family version.
        artifact_kind:
            Artifact type produced by the prompt.
        instructions:
            Stable task-specific instructions before evidence assembly.
        prompt_hash:
            SHA-256 hash of the exact stable prompt definition.
    """

    prompt_id: str = Field(min_length=1, description="Stable prompt identifier.")
    artifact_kind: ArtifactKind = Field(description="Artifact type produced by this prompt.")
    instructions: str = Field(
        min_length=1,
        description="Stable task-specific prompt instructions.",
    )

    @computed_field(description="SHA-256 hash of the stable prompt definition.")  # type: ignore[prop-decorator]
    @property
    def prompt_hash(self) -> str:
        """Calculate the stable prompt-template hash.

        Returns:
            Lowercase hexadecimal SHA-256 digest.
        """

        return hashlib.sha256(prompt_template(self).encode("utf-8")).hexdigest()


COMPONENT_PROMPT = PromptDefinition(
    prompt_id=COMPONENT_PROMPT_ID,
    artifact_kind="component",
    instructions=COMPONENT_PROMPT,
)
OVERVIEW_PROMPT = PromptDefinition(
    prompt_id=OVERVIEW_PROMPT_ID,
    artifact_kind="overview",
    instructions=OVERVIEW_PROMPT,
)
ARCHITECTURE_PROMPT = PromptDefinition(
    prompt_id=ARCHITECTURE_PROMPT_ID,
    artifact_kind="architecture",
    instructions=ARCHITECTURE_PROMPT,
)
TESTING_OPERATIONS_PROMPT = PromptDefinition(
    prompt_id=TESTING_OPERATIONS_PROMPT_ID,
    artifact_kind="testing_operations",
    instructions=TESTING_OPERATIONS_PROMPT,
)
PROMPTS = (
    ARCHITECTURE_PROMPT,
    COMPONENT_PROMPT,
    OVERVIEW_PROMPT,
    TESTING_OPERATIONS_PROMPT
)


def prompt_template(prompt: PromptDefinition) -> str:
    specification = SUMMARY_SPECIFICATIONS[prompt.artifact_kind]
    skeleton_parts = []
    for section in specification.sections:
        skeleton_parts.append(
            f"## {section.heading}\n\n<concise factual prose>\n\n"
            "### Evidence\n\n- <exact path from allowed catalog>"
        )
    skeleton = "\n\n".join(skeleton_parts)

    return PROMPT_TEMPLATE.format(
        safety_prompt=SAFETY_PROMPT,
        prompt=prompt.instructions,
        skeleton=skeleton,
        example_heading=specification.sections[0].heading,
    )
