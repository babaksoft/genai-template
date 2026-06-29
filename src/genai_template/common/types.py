from enum import StrEnum


class VectorDistance(StrEnum):
    """Supported vector distance metrics."""

    COSINE = "cosine"
    L2 = "l2"
    INNER_PRODUCT = "ip"
