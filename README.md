# Grounded Prediction Networks

> **A Single-Layer Model Can Do Language Modeling.**
> Companion code for the paper.

A Grounded Prediction Network (GPN) is a language model built around one
state vector and a single recurrent block — one FFN and one shared matrix
memory, revisited at every step. There is no per-layer state and no stack;
depth comes from time.

At 130M parameters, a 1-layer GPN+M reaches FineWeb-Edu PPL **18.06**,
within **13%** of a 12-layer Transformer++ (16.05) and **18%** of a
10-layer Gated DeltaNet (15.34). A 2-layer variant brings the gap to
**6%/11%**.

This is a research result about what a single shallow recurrent layer can
carry, not a production-ready system. Training is sequential across time
in Python and is substantially more expensive per step than the
transformer/GDN baselines.

## The loop

```
s_pred[t]   = Predict( s_ground[t-1] )         # FFN + matrix-memory read
s_ground[t] = Ground ( s_pred[t-1], x[t] )     # fuse the observed token
L[t]        = CE( Decode(s_pred[t]), x[t+1] )
```

Conceptually, the state's prediction of the next step can be viewed as
part of the encoding of that step — a single vector plays both roles,
with grounding completing the encoding.

## Layout

```
src/
├── data/                     memory-mapped TokenDataset
├── jax/                      GPN / GPN+M model (JAX/Flax)
│   ├── modules/
│   ├── models/{sot, sot_m}/
│   └── training/
└── torch/                    baselines (PyTorch)
    ├── modules/
    ├── models/{transformer, gdn}/
    └── training/

tests/                        pytest tree mirroring src/
scripts/data/                 tokenizer + shard prep
paper/v1-arxiv/               LaTeX source + compiled PDF
```

The model is named `sot` / `sot_m` in the code for historical reasons
(an earlier project name); the paper refers to the same model as
GPN / GPN+M.

## Quick start

```bash
make setup                                                                 # venv + deps

# Train (GPN / GPN+M)
PYTHONPATH=. python -m src.jax.models.sot.train   --config <cfg>.json
PYTHONPATH=. python -m src.jax.models.sot_m.train --config <cfg>.json

# Train baselines
PYTHONPATH=. python -m src.torch.models.transformer.train --config <cfg>.json
PYTHONPATH=. python -m src.torch.models.gdn.train         --config <cfg>.json

# Tests
PYTHONPATH=. python -m pytest tests/ -v
```

## Paper

`paper/v1-arxiv/main.pdf` — the compiled paper. LaTeX source alongside.

## Citation

```bibtex
@misc{wang2026gpn,
  author        = {Zanmin Wang},
  title         = {A Single-Layer Model Can Do Language Modeling},
  year          = {2026},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

## License

MIT — see `LICENSE`.
