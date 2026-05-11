"""Download and tokenize a HuggingFace dataset.

Tokenizes directly on Arrow data using datasets.map() for parallelism,
then writes document-aligned shards.

Output format:
  data_XXXX.bin        — flat uint32 token shards
  data_XXXX.index.json — per-shard JSON array of document token lengths
  meta.json            — {tokenizer, vocab_size, total_tokens, total_docs, num_shards}

Usage:
    python scripts/data/download_n_tokenize.py --config <path_to_config.json>
"""

import argparse
import json
import os
from multiprocessing import cpu_count

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


DEFAULT_SHARD_SIZE = 500_000_000  # 500M tokens per output shard


def _write_shard(buffer, shard_idx, output_dir):
    arr = np.array(buffer, dtype=np.uint32)
    path = os.path.join(output_dir, f"data_{shard_idx:04d}.bin")
    arr.tofile(path)


def _write_index(doc_lengths, shard_idx, output_dir):
    path = os.path.join(output_dir, f"data_{shard_idx:04d}.index.json")
    with open(path, "w") as f:
        json.dump(doc_lengths, f)


def prepare(config):
    dataset_name = config["dataset"]
    name = config["name"]
    split = config["split"]
    tokenizer_name = config["tokenizer"]
    output_dir = config["output_dir"]
    num_proc = config.get("num_proc", cpu_count())
    shard_size = config.get("shard_size", DEFAULT_SHARD_SIZE)

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.model_max_length = int(1e12)
    eos_id = tokenizer.eos_token_id
    vocab_size = tokenizer.vocab_size

    print("=========================================")
    print(" Download & Tokenize")
    print("=========================================")
    print(f"  Dataset:    {dataset_name} ({name})")
    print(f"  Split:      {split}")
    print(f"  Tokenizer:  {tokenizer_name} (vocab {vocab_size})")
    print(f"  Output:     {output_dir}")
    print(f"  Num proc:   {num_proc}")
    print(f"  Shard size: {shard_size:,} tokens ({shard_size * 4 / 1e9:.1f} GB)")
    print()

    print("[1/3] Loading dataset...")
    ds = load_dataset(dataset_name, name=name, split=split, num_proc=num_proc)
    print(f"  Loaded {len(ds):,} documents.")

    def tokenize_fn(batch):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        encoded = tokenizer(batch["text"])["input_ids"]
        for ids in encoded:
            ids.append(eos_id)
        return {"input_ids": encoded}

    print("\n[2/3] Tokenizing...")
    ds = ds.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )

    print("\n[3/3] Writing shards...")
    buffer = []
    doc_lengths = []
    shard_idx = 0
    total_tokens = 0
    total_docs = 0

    for example in tqdm(ds, desc="Writing shards", unit="doc"):
        tokens = example["input_ids"]
        buffer.extend(tokens)
        doc_lengths.append(len(tokens))
        total_tokens += len(tokens)
        total_docs += 1

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
        "total_docs": total_docs,
        "num_shards": shard_idx,
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone: {total_tokens:,} tokens, {total_docs:,} docs, {shard_idx} shards")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    prepare(cfg)
