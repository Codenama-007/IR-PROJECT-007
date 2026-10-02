"""Retrieval accuracy metrics. All take lists of chunk ids."""


def precision_at_k(retrieved_ids: list, relevant_ids: set) -> float:
    """Fraction of the retrieved chunks that are relevant."""
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for cid in retrieved_ids if cid in relevant_ids)
    return hits / len(retrieved_ids)


def reciprocal_rank(retrieved_ids: list, relevant_ids: set) -> float:
    """1 / rank of the first relevant chunk (0 if none was retrieved)."""
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def overlap(ids_a: list, ids_b: list) -> set:
    """Chunk ids returned by both retrievers."""
    return set(ids_a) & set(ids_b)