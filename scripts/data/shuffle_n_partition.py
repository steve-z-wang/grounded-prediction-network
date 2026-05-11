"""Partition tokenized data into non-overlapping buckets.

Reads document-aligned shards, shuffles documents, and splits into buckets
via deficit-based sampling. Each bucket gets its own set of shards.

Usage:
    python scripts/data/shuffle_n_partition.py --config <path_to_config.json>
"""

import argparse
import json
import os
from collections import namedtuple

import numpy as np
from tqdm import tqdm


DocRef = namedtuple("DocRef", ["shard_idx", "offset", "length"])

DEFAULT_SHARD_SIZE = 500_000_000  # 500M tokens per output shard


def _load_doc_refs(input_dir):
    shard_idx = 0
    doc_refs = []
    while True:
        index_path = os.path.join(input_dir, f"data_{shard_idx:04d}.index.json")
        if not os.path.exists(index_path):
            break
        with open(index_path) as f:
            lengths = json.load(f)
        offset = 0
        for length in lengths:
            doc_refs.append(DocRef(shard_idx=shard_idx, offset=offset, length=length))
            offset += length
        shard_idx += 1
    return doc_refs, shard_idx


def _read_doc_tokens(doc_ref, shards):
    shard = shards[doc_ref.shard_idx]
    return np.array(shard[doc_ref.offset : doc_ref.offset + doc_ref.length])


def _write_shard(buffer, shard_idx, output_dir):
    arr = np.array(buffer, dtype=np.uint32)
    path = os.path.join(output_dir, f"data_{shard_idx:04d}.bin")
    arr.tofile(path)


def _write_index(doc_lengths, shard_idx, output_dir):
    path = os.path.join(output_dir, f"data_{shard_idx:04d}.index.json")
    with open(path, "w") as f:
        json.dump(doc_lengths, f)


def _write_bucket(bucket_docs, shards, output_dir, tokenizer_name, vocab_size,
                   shard_size=DEFAULT_SHARD_SIZE):
    os.makedirs(output_dir, exist_ok=True)

    buffer = []
    doc_lengths = []
    shard_idx = 0
    total_tokens = 0

    for doc_ref in tqdm(bucket_docs, desc=f"Writing {os.path.basename(output_dir)}"):
        tokens = _read_doc_tokens(doc_ref, shards)
        buffer.extend(tokens)
        doc_lengths.append(len(tokens))
        total_tokens += len(tokens)

        if len(buffer) >= shard_size:
            _write_shard(buffer, shard_idx, output_dir)
            _write_index(doc_lengths, shard_idx, output_dir)
            buffer = []
            doc_lengths = []
            shard_idx += 1

    if buffer:
        _write_shard(buffer, shard_idx, output_dir)
        _write_index(doc_lengths, shard_idx, output_dir)
        shard_idx += 1

    meta = {
        "tokenizer": tokenizer_name,
        "vocab_size": vocab_size,
        "total_tokens": total_tokens,
        "total_docs": len(bucket_docs),
        "num_shards": shard_idx,
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return total_tokens


def partition(config):
    input_dir = config["input_dir"]
    output_dir = config["output_dir"]
    seed = config["seed"]
    buckets_cfg = config["buckets"]
    shard_size = config.get("shard_size", DEFAULT_SHARD_SIZE)

    with open(os.path.join(input_dir, "meta.json")) as f:
        meta = json.load(f)
    tokenizer_name = meta["tokenizer"]
    vocab_size = meta["vocab_size"]

    print("=========================================")
    print(" Shuffle & Partition")
    print("=========================================")
    print(f"  Input:      {input_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Seed:       {seed}")
    print(f"  Buckets:    {len(buckets_cfg)}")
    print(f"  Tokenizer:  {tokenizer_name} (vocab {vocab_size})")
    print(f"  Shard size: {shard_size:,} tokens ({shard_size * 4 / 1e9:.1f} GB)")
    print()

    print("[1/5] Loading document index...")
    doc_refs, num_shards = _load_doc_refs(input_dir)
    print(f"  Loaded {len(doc_refs):,} documents from {num_shards} shards.")

    print(f"\n[2/5] Memory-mapping shards...")
    shards = []
    for i in range(num_shards):
        path = os.path.join(input_dir, f"data_{i:04d}.bin")
        shards.append(np.memmap(path, dtype=np.uint32, mode="r"))
    print(f"  Mapped {num_shards} shards.")

    print(f"\n[3/5] Shuffling {len(doc_refs):,} documents (seed={seed})...")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(doc_refs))
    print(f"  Shuffled.")

    bucket_names = [b["name"] for b in buckets_cfg]
    bucket_targets = [b["tokens"] for b in buckets_cfg]
    bucket_current = [0] * len(buckets_cfg)
    bucket_docs = [[] for _ in buckets_cfg]

    total_target = sum(bucket_targets)
    print(f"\n[4/5] Allocating docs to {len(buckets_cfg)} bucket(s)...")
    print(f"  Target: {total_target:,} tokens total")
    for b in buckets_cfg:
        print(f"    {b['name']}: {b['tokens']:,} tokens")

    for idx in tqdm(indices, desc="  Allocating"):
        doc = doc_refs[idx]
        deficits = [t - c for t, c in zip(bucket_targets, bucket_current)]
        best = max(range(len(deficits)), key=lambda i: deficits[i])
        if deficits[best] <= 0:
            break
        bucket_docs[best].append(doc)
        bucket_current[best] += doc.length

    print(f"\n[5/5] Writing buckets to disk...")
    single_bucket = len(bucket_names) == 1
    for i, name in enumerate(bucket_names):
        # If only one bucket, write directly to output_dir (skip subfolder)
        bucket_dir = output_dir if single_bucket else os.path.join(output_dir, name)
        print(f"  Writing bucket '{name}' ({len(bucket_docs[i]):,} docs)...")
        actual = _write_bucket(
            bucket_docs[i], shards, bucket_dir, tokenizer_name, vocab_size,
            shard_size=shard_size,
        )
        pct = actual / bucket_targets[i] * 100 if bucket_targets[i] > 0 else 0
        print(
            f"  Bucket '{name}': {actual:,} tokens "
            f"({pct:.1f}% of target {bucket_targets[i]:,})"
        )

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    partition(cfg)
