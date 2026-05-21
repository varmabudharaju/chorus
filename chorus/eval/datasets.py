"""Dataset loading + partitioning for the eval harness.

Wraps HuggingFace `datasets` for the common loading patterns and implements
IID + Dirichlet-based non-IID partitioning into per-client splits.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def load_dataset_for_eval(
    name: str,
    split: str = "train",
    max_examples: int | None = None,
    text_field: str = "text",
    label_field: str = "label",
    config_name: str | None = None,
) -> list[dict[str, Any]]:
    """Load a HuggingFace dataset and return a list of {text_field, label_field, ...} dicts.

    Args:
        name: HF dataset name (e.g., "glue", "wikitext").
        split: Which split to load (e.g., "train", "validation").
        max_examples: If set, truncate to the first N examples.
        text_field: Column name for the text input.
        label_field: Column name for the label (may not exist for LM tasks).
        config_name: Optional HF config name (e.g., "sst2" for "glue").

    Returns:
        List of dicts; each dict has the original columns.
    """
    from datasets import load_dataset

    ds = load_dataset(name, config_name, split=split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))
    return [dict(ex) for ex in ds]


def partition_iid(
    dataset: list[dict[str, Any]],
    num_clients: int,
    seed: int = 0,
) -> list[list[dict[str, Any]]]:
    """Shuffle the dataset and split into `num_clients` near-equal partitions.

    Returns a list of `num_clients` lists. Partitions sum to len(dataset)
    and no example appears in more than one partition. Sizes differ by at most 1.
    """
    rng = np.random.default_rng(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    partitions: list[list[dict[str, Any]]] = [[] for _ in range(num_clients)]
    for i, idx in enumerate(indices):
        partitions[i % num_clients].append(dataset[idx])
    return partitions


def partition_non_iid_dirichlet(
    dataset: list[dict[str, Any]],
    num_clients: int,
    alpha: float = 0.5,
    label_field: str = "label",
    seed: int = 0,
) -> list[list[dict[str, Any]]]:
    """Partition by drawing per-client class proportions from Dir(alpha).

    Low `alpha` (e.g., 0.1) produces highly skewed per-client distributions
    (each client sees only a few classes). High `alpha` (e.g., 100) approaches
    IID.

    Args:
        dataset: List of dicts with a label field.
        num_clients: Number of partitions.
        alpha: Dirichlet concentration parameter.
        label_field: Dict key holding the class label.
        seed: RNG seed.

    Returns:
        List of `num_clients` lists.
    """
    rng = np.random.default_rng(seed)

    # Group example indices by class
    by_class: dict[int, list[int]] = {}
    for i, ex in enumerate(dataset):
        cls = ex[label_field]
        by_class.setdefault(cls, []).append(i)

    partitions: list[list[dict[str, Any]]] = [[] for _ in range(num_clients)]
    for cls, idxs in by_class.items():
        rng.shuffle(idxs)
        # Draw client proportions from Dirichlet
        proportions = rng.dirichlet([alpha] * num_clients)
        # Compute split points
        cuts = (np.cumsum(proportions) * len(idxs)).astype(int)
        cuts[-1] = len(idxs)  # guard against float truncation dropping the last item(s)
        prev = 0
        for c, cut in enumerate(cuts):
            for j in idxs[prev:cut]:
                partitions[c].append(dataset[j])
            prev = cut

    return partitions
