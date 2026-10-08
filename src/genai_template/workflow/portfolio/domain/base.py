from pydantic import BaseModel, ConfigDict


class ImmutableDomainModel(BaseModel):
    """Base model for immutable values crossing workflow boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True)
