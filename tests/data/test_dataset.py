import json
import os

import numpy as np
import pytest

from src.data import TokenDataset


@pytest.fixture
def tmp_dataset(tmp_path):
    """Create a small in-memory dataset: 2 shards of 100 uint32 tokens each."""
    d = tmp_path / "data"
    d.mkdir()
    for i in range(2):
        arr = np.arange(i * 100, (i + 1) * 100, dtype=np.uint32)
        arr.tofile(d / f"data_{i:04d}.bin")
    return str(d)


def test_dataset_loads(tmp_dataset):
    ds = TokenDataset(tmp_dataset, chunk_size=20)
    assert ds.total_tokens == 200
    assert ds.n_chunks == 10


def test_dataset_batches_shape(tmp_dataset):
    ds = TokenDataset(tmp_dataset, chunk_size=10)
    batches = list(ds.batches(batch_size=4))
    assert len(batches) == 5
    for b in batches:
        assert b.shape == (4, 10)


def test_dataset_batches_start(tmp_dataset):
    ds = TokenDataset(tmp_dataset, chunk_size=10)
    all_batches = list(ds.batches(batch_size=2))
    resumed = list(ds.batches(batch_size=2, start=5))
    # resumed should be the tail of all_batches
    assert len(resumed) == len(all_batches) - 5
    for a, b in zip(all_batches[5:], resumed):
        assert (a == b).all()


def test_dataset_cross_shard(tmp_dataset):
    """chunk_size that spans two shards must still read contiguously."""
    ds = TokenDataset(tmp_dataset, chunk_size=150)  # > shard size (100)
    batches = list(ds.batches(batch_size=1))
    assert len(batches) == 1
    # Should contain 0..149
    assert batches[0][0, 0].item() == 0
    assert batches[0][0, -1].item() == 149
