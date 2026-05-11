"""Token dataset: memory-mapped .bin shards."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np


class BaseDataset(ABC):
    """Interface for token datasets."""

    total_tokens: int
    chunk_size: int

    @abstractmethod
    def __init__(self, data_dir: str, chunk_size: int) -> None: ...

    @abstractmethod
    def batches(self, batch_size: int, start: int = 0) -> Iterator[np.ndarray]:
        """Yield batches starting from batch index `start`."""
        ...


class TokenDataset(BaseDataset):
    """Non-overlapping windows of token IDs from memory-mapped .bin shards."""

    def __init__(self, data_dir: str, chunk_size: int) -> None:
        data_dir = str(data_dir)
        shard_paths = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.startswith("data_") and f.endswith(".bin")
        )
        if not shard_paths:
            raise FileNotFoundError(f"No data_*.bin shards found in {data_dir}")

        self.shards = [np.memmap(p, dtype=np.uint32, mode="r") for p in shard_paths]
        self.shard_sizes = [len(s) for s in self.shards]
        self.total_tokens = sum(self.shard_sizes)

        self.shard_offsets = np.zeros(len(self.shards) + 1, dtype=np.int64)
        for i, size in enumerate(self.shard_sizes):
            self.shard_offsets[i + 1] = self.shard_offsets[i] + size

        self.chunk_size = chunk_size
        self.n_chunks = self.total_tokens // chunk_size
        print(
            f"Loaded {len(shard_paths)} shards, {self.total_tokens:,} tokens "
            f"(memory-mapped)"
        )

    def _get_tokens(self, start: int, end: int) -> np.ndarray:
        for i in range(len(self.shards)):
            s_start = self.shard_offsets[i]
            s_end = self.shard_offsets[i + 1]
            if start >= s_start and end <= s_end:
                return np.array(self.shards[i][start - s_start : end - s_start])

        parts: list[np.ndarray] = []
        for i in range(len(self.shards)):
            s_start = self.shard_offsets[i]
            s_end = self.shard_offsets[i + 1]
            if start >= s_end:
                continue
            if end <= s_start:
                break
            local_start = max(0, start - s_start)
            local_end = min(s_end - s_start, end - s_start)
            parts.append(np.array(self.shards[i][local_start:local_end]))
        return np.concatenate(parts)

    def _get_batch(self, indices: np.ndarray) -> np.ndarray:
        batch = np.empty((len(indices), self.chunk_size), dtype=np.int32)
        for i, idx in enumerate(indices):
            start = idx * self.chunk_size
            batch[i] = self._get_tokens(start, start + self.chunk_size)
        return batch

    def batches(self, batch_size: int, start: int = 0) -> Iterator[np.ndarray]:
        """Yield sequential batches starting from batch index `start`.

        A background thread prefetches the next batch while the current one
        is consumed.
        """
        total_batches = self.n_chunks // batch_size
        if start >= total_batches:
            return

        def fetch(batch_idx: int) -> np.ndarray:
            chunk_start = batch_idx * batch_size
            indices = np.arange(chunk_start, chunk_start + batch_size)
            return self._get_batch(indices)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(fetch, start)
            next_idx = start + 1

            for _ in range(start, total_batches):
                numpy_batch = future.result()
                if next_idx < total_batches:
                    future = pool.submit(fetch, next_idx)
                    next_idx += 1
                yield numpy_batch
