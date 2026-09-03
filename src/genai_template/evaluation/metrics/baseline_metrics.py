"""Baseline retrieval metrics for evaluating retrieval performance."""


def calculate_hit_at_k(
    retrieved_documents: list[str],
    expected_documents: list[str],
    k: int,
) -> bool:
    """Calculate whether at least one relevant document appears in top K.

    Args:
        retrieved_documents:
            Document names returned by retrieval in ranked order.

        expected_documents:
            Document names considered relevant.

        k:
            Number of top-ranked documents to evaluate.

    Returns:
        True when at least one relevant document appears in top K.
    """

    retrieved = set(retrieved_documents[:k])
    expected = set(expected_documents)

    return bool(retrieved & expected)


def calculate_recall_at_k(
    retrieved_documents: list[str],
    expected_documents: list[str],
    k: int,
) -> float:
    """Calculate retrieval recall at K.

    Args:
        retrieved_documents:
            Document names returned by retrieval in ranked order.

        expected_documents:
            Document names considered relevant.

        k:
            Number of top-ranked documents to evaluate.

    Returns:
        Proportion of relevant documents retrieved in top K.
    """

    expected = set(expected_documents)
    if not expected:
        return 0.0

    retrieved = set(retrieved_documents[:k])

    return len(retrieved & expected) / len(expected)


def calculate_precision_at_k(
    retrieved_documents: list[str],
    expected_documents: list[str],
    k: int,
) -> float:
    """Calculate retrieval precision at K.

    Args:
        retrieved_documents:
            Document names returned by retrieval in ranked order.

        expected_documents:
            Document names considered relevant.

        k:
            Number of top-ranked documents to evaluate.

    Returns:
        Proportion of top-K retrieved documents that are relevant.
    """

    retrieved = retrieved_documents[:k]
    if not retrieved:
        return 0.0

    expected = set(expected_documents)

    return sum(document in expected for document in retrieved) / len(retrieved)
