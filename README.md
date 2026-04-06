# Spectral Index — KV Cache Effective Dimensionality Leaderboard

**How many dimensions does an LLM's KV cache actually use?**

This repository measures the *effective dimensionality* (d_eff) of key and
value cache representations across open-source LLMs, and publishes the results
as a public leaderboard.

**Live leaderboard:** https://spectral-index.github.io  
**Paper:** [SpectralQuant: Quantizing the KV Cache via Spectral Analysis](https://arxiv.org/abs/2405.20582)

---

## What is d_eff and why does it matter?

When a transformer generates a token, it stores a *key* vector and a *value*
vector for every layer and attention head — collectively the **KV cache**.
For a long sequence or a large model these tensors dominate GPU memory.

A central insight from the SpectralQuant paper is that the empirical
distribution of KV vectors is far from isotropic: **most of the variance is
concentrated in a tiny subspace**.  The *participation ratio*

```
d_eff = (Σ λᵢ)² / Σ(λᵢ²)
```

quantifies this.  If the covariance eigenspectrum is flat, d_eff equals
the full head dimension (e.g. 128).  If all energy is in one direction,
d_eff = 1.  Measured values for real LLMs cluster in a startling range:

| Cache | Typical d_eff | Meaning |
|-------|--------------|---------|
| Keys  | **3–6**      | Only 3–6 out of 128 dimensions carry signal |
| Values| **38–110**   | Much higher, but still far below head_dim |

**Why this matters for compression:**  Low d_eff means KV data can be
projected to a smaller basis, quantized more aggressively, or pruned —
all with minimal accuracy loss.  d_eff is therefore a principled bound on
how compressible a model's KV cache is, without running any downstream
benchmarks.

The companion metric **κ (kappa)** — the ratio of the last "in-subspace"
eigenvalue to the first "out-of-subspace" eigenvalue — measures the
sharpness of the spectral cliff.  Large κ means a cleaner low-rank structure.

---

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/<your-org>/spectral-index
cd spectral-index
pip install -e ".[dev]"

# 2. Measure a model (public model, no token needed)
python -m src.measure \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --n-sequences 100 \
  --max-length 512 \
  --output results/qwen2.5-7b.json \
  --device cuda

# 3. Rebuild the leaderboard data file
python scripts/build_leaderboard.py
```

That's it.  The result JSON is saved to `results/` and is ready to commit.

---

## Measuring any HuggingFace model

```bash
python -m src.measure --model <HF_MODEL_ID> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | HuggingFace model ID or local path |
| `--n-sequences` | `100` | Number of WikiText-103 calibration sequences |
| `--max-length` | `512` | Token length per sequence |
| `--output` | `results/<slug>.json` | Output JSON file path |
| `--device` | `auto` | `cuda`, `cpu`, `mps`, or `auto` |
| `--seed` | `42` | RNG seed for reproducibility |
| `--verbose` | off | Enable DEBUG logging |

### Gated models (Llama, Gemma, …)

Export your HuggingFace token before running:

```bash
export HF_TOKEN=hf_...
python -m src.measure --model "meta-llama/Llama-3.1-8B-Instruct" --device cuda
```

### Batch measurement (all target models)

```bash
python scripts/run_all.py --device cuda
```

Already-measured models are skipped automatically.  Interrupt and restart
at any time — progress is preserved.

---

## Running on Modal (cloud GPU)

For large models (70B+) that require an A100, use the Modal runner:

### Prerequisites

```bash
pip install modal
modal setup                                          # authenticate
modal secret create huggingface-secret HF_TOKEN=<your_token>
```

### Measure a single model

```bash
modal run scripts/modal_run.py::measure_single \
    --model-name "meta-llama/Llama-3.1-70B-Instruct"
```

### Measure all models in parallel

```bash
modal run scripts/modal_run.py --n-sequences 100
```

Each model runs in its own A100-80GB container.  Results are downloaded
locally to `results/` when complete.

### (Optional) Persist results to a Modal Volume

Uncomment the `results_volume` lines in `scripts/modal_run.py` and create
the volume:

```bash
modal volume create spectral-index-results
```

---

## Output format

Each measurement produces a JSON file with this structure:

```jsonc
{
  "schema_version": "1.0",
  "provenance": {
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "n_sequences": 100,
    "max_length": 512,
    "measured_at": "2025-01-15T00:00:00+00:00",
    ...
  },
  "architecture": {
    "n_layers": 32,
    "n_kv_heads": 8,
    "head_dim": 128
  },
  "aggregate": {
    "key_deff": 3.64,       // participation ratio for keys
    "key_kappa": 1.257,     // spectral gap for keys
    "key_dims95": 7,        // dims needed to explain 95% of key variance
    "key_dims99": 14,
    "val_deff": 44.21,
    "val_kappa": 1.031,
    "val_dims95": 62,
    "val_dims99": 89
  },
  "per_layer": {
    "key_deff":  [3.375, 3.375, ...],   // one entry per layer
    "key_kappa": [1.499, 1.365, ...],
    "val_deff":  [...],
    "val_kappa": [...]
  },
  "representative_spectra": {
    "key_eigenvalues": [...],   // first 32 eigenvalues (descending)
    "val_eigenvalues": [...]
  }
}
```

---

## Contributing measurements

New measurements are very welcome!  To add a model:

1. Run `python -m src.measure --model <YOUR_MODEL>` (or the Modal runner).
2. The JSON file will be saved to `results/`.
3. Open a pull request with just the new `results/<model>.json` file.

**Guidelines:**
- Use the default settings (n=100, max_length=512, seed=42) for consistency.
- Include `"source"` in the provenance if the numbers come from a paper.
- Do not modify existing verified result files.

---

## Methodology

### Calibration data

We use the WikiText-103 validation split, chunked into non-overlapping
512-token windows.  100 evenly-spaced windows are selected for coverage.
This is the same corpus used in the SpectralQuant experiments.

### Covariance estimation

For every (layer, KV-head) pair, we accumulate the **uncentered** second-moment
matrix across all calibration tokens:

```
C = (Σ xᵢ xᵢᵀ) / n     where xᵢ ∈ ℝ^{head_dim}
```

Accumulation is done in float64 for numerical stability.

### Eigendecomposition

```python
ev, _ = torch.linalg.eigh(C)
ev = ev.flip(0).clamp(min=0)   # descending, non-negative
```

### Metrics

```python
d_eff  = (ev.sum()**2 / (ev**2).sum()).item()
kappa  = ev[round(d_eff)-1] / ev[round(d_eff)]
dims_k = (ev.cumsum(0) / ev.sum() >= 0.95).nonzero()[0] + 1
```

### GQA support

Models with grouped-query attention (n_kv_heads < n_heads) are handled
automatically — only the true KV heads are measured, not the expanded copies.

---

## Repository layout

```
spectral-index/
├── src/
│   ├── __init__.py
│   └── measure.py          ← core measurement script
├── scripts/
│   ├── run_all.py          ← batch runner (all target models)
│   ├── modal_run.py        ← Modal A100 runner
│   └── build_leaderboard.py← aggregates results → site/data.json
├── results/                ← per-model JSON files (committed to git)
│   ├── llama-3.1-8b.json
│   └── ...
├── site/                   ← leaderboard website (see sister repo)
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Citation

If you use this data or code in your research, please cite the SpectralQuant
paper on which the methodology is based:

```bibtex
@article{spectralquant2024,
  title   = {SpectralQuant: Quantizing the KV Cache via Spectral Analysis},
  year    = {2024},
  url     = {https://arxiv.org/abs/2405.20582}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
