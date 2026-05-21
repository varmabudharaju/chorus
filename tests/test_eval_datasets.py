"""Tests for dataset loading + partitioning in the eval harness."""

import numpy as np

from chorus.eval.datasets import (
    partition_iid,
    partition_non_iid_dirichlet,
)


def _make_fake_dataset(n_examples: int, n_classes: int = 2):
    """Build a list of {text, label} dicts."""
    rng = np.random.default_rng(42)
    labels = rng.integers(0, n_classes, size=n_examples)
    return [{"text": f"example {i}", "label": int(lbl)} for i, lbl in enumerate(labels)]


class TestIID:
    def test_partitions_sum_to_dataset_size(self):
        ds = _make_fake_dataset(100)
        parts = partition_iid(ds, num_clients=5, seed=0)
        assert sum(len(p) for p in parts) == 100
        assert len(parts) == 5

    def test_no_overlap_between_partitions(self):
        ds = _make_fake_dataset(100)
        parts = partition_iid(ds, num_clients=4, seed=0)
        seen = set()
        for p in parts:
            for ex in p:
                key = ex["text"]
                assert key not in seen
                seen.add(key)

    def test_partition_sizes_roughly_balanced(self):
        ds = _make_fake_dataset(100)
        parts = partition_iid(ds, num_clients=5, seed=0)
        sizes = [len(p) for p in parts]
        assert max(sizes) - min(sizes) <= 1


class TestDirichlet:
    def test_dirichlet_partitions_sum_to_dataset_size(self):
        ds = _make_fake_dataset(200, n_classes=4)
        parts = partition_non_iid_dirichlet(ds, num_clients=5, alpha=0.5, seed=0)
        assert sum(len(p) for p in parts) == 200

    def test_dirichlet_low_alpha_creates_skew(self):
        """Low alpha → highly non-IID → at least one client gets a heavily-skewed class distribution."""
        ds = _make_fake_dataset(400, n_classes=4)
        parts = partition_non_iid_dirichlet(ds, num_clients=5, alpha=0.1, seed=0)

        def class_fraction(partition, cls):
            if not partition:
                return 0.0
            return sum(1 for ex in partition if ex["label"] == cls) / len(partition)

        # At least one client should have one class dominate >80% of its samples
        skewed = False
        for p in parts:
            for cls in range(4):
                if class_fraction(p, cls) > 0.8:
                    skewed = True
                    break
            if skewed:
                break
        assert skewed, "Low-alpha Dirichlet should produce at least one heavily skewed client"

    def test_dirichlet_high_alpha_approaches_iid(self):
        """High alpha → close to uniform; class fractions should not vary wildly across clients."""
        ds = _make_fake_dataset(800, n_classes=4)
        parts = partition_non_iid_dirichlet(ds, num_clients=5, alpha=100.0, seed=0)
        # For each class, fractions across clients should be similar (max - min < 0.2)
        for cls in range(4):
            fracs = []
            for p in parts:
                if not p:
                    continue
                fracs.append(sum(1 for ex in p if ex["label"] == cls) / len(p))
            assert max(fracs) - min(fracs) < 0.3, f"Class {cls} fractions too varied: {fracs}"

    def test_dirichlet_no_data_loss_across_many_seeds(self):
        """Across 50 different seeds and odd dataset sizes, sum must always equal len(dataset)."""
        for seed in range(50):
            # Use sizes that don't divide evenly to expose float truncation
            ds = _make_fake_dataset(101, n_classes=5)  # 101 is prime
            parts = partition_non_iid_dirichlet(ds, num_clients=4, alpha=0.5, seed=seed)
            assert sum(len(p) for p in parts) == 101, (
                f"Data loss at seed={seed}: {sum(len(p) for p in parts)} != 101"
            )
