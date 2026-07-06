"""Database models."""

from genai_template.db.models.experiment import Experiment
from genai_template.db.models.run import Run

__all__ = [
    "Experiment",
    "Run",
]
