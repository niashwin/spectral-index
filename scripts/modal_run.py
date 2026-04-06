"""
modal_run.py — Modal runner for GPU-accelerated Spectral Index measurements.

Each model is measured inside a Modal A100-80GB container.  Results are
returned as Python dicts and can optionally be pushed to a Modal Volume for
persistent storage.

Quickstart
----------
    # Measure a single model
    modal run scripts/modal_run.py::measure_single \\
        --model-name "meta-llama/Llama-3.1-8B-Instruct"

    # Measure all models in the batch list
    modal run scripts/modal_run.py::measure_batch

    # Deploy as a persistent app
    modal deploy scripts/modal_run.py

Prerequisites
-------------
1.  Install Modal:      pip install modal
2.  Authenticate:       modal setup
3.  Create HF secret:   modal secret create huggingface-secret HF_TOKEN=<your_token>
4.  (Optional) Create a volume for results:
        modal volume create spectral-index-results
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# App + Image
# ---------------------------------------------------------------------------

app = modal.App("spectral-index")

# The measurement image bundles everything needed to run src/measure.py.
# We mount the local src/ package so Modal gets the latest version.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "accelerate>=0.29.0",
        "numpy>=1.26.0",
    )
    # Make the local src package available inside the container
    .add_local_python_source("src")
)

# Optional: persistent volume to accumulate results across runs.
# Create it with: modal volume create spectral-index-results
# Then uncomment the volume= line in the @app.function decorators below.
#
# results_volume = modal.Volume.from_name("spectral-index-results", create_if_missing=True)
# VOLUME_MOUNT = "/results"

# ---------------------------------------------------------------------------
# HuggingFace secret (set HF_TOKEN in Modal's secrets dashboard)
# ---------------------------------------------------------------------------
HF_SECRET = modal.Secret.from_name("huggingface-secret")

# ---------------------------------------------------------------------------
# Core measurement function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=modal.gpu.A100(count=1, size="80GB"),
    timeout=3600,          # 1-hour wall-clock limit per model
    memory=65536,          # 64 GiB RAM
    secrets=[HF_SECRET],
    # volumes={VOLUME_MOUNT: results_volume},  # uncomment to persist results
)
def measure_model(
    model_name: str,
    n_sequences: int = 100,
    max_length: int = 512,
    seed: int = 42,
    save_to_volume: bool = False,
) -> dict:
    """
    Measure effective KV-cache dimensionality for *model_name* on an A100.

    Parameters
    ----------
    model_name      HuggingFace model ID, e.g. "meta-llama/Llama-3.1-8B-Instruct"
    n_sequences     Number of WikiText-103 calibration sequences
    max_length      Token length per sequence
    seed            RNG seed for reproducibility
    save_to_volume  If True, write results to the attached Modal Volume

    Returns
    -------
    Full result dict (same schema as results/*.json)
    """
    import os

    from src.measure import _setup_logging, measure

    _setup_logging(verbose=False)

    output_path = None
    if save_to_volume:
        # results_volume is mounted at VOLUME_MOUNT
        slug = model_name.replace("/", "--").replace(" ", "_").lower()
        output_path = Path(f"/results/{slug}.json")

    result = measure(
        model_name=model_name,
        n_sequences=n_sequences,
        max_length=max_length,
        output_path=output_path,
        device_str="cuda",
        hf_token=os.environ.get("HF_TOKEN"),
        seed=seed,
    )

    if save_to_volume and output_path:
        # Commit writes so they are visible outside the container
        # results_volume.commit()
        pass

    return result


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

# Mirror of the MODELS list in scripts/run_all.py
MODELS: list[tuple[str, bool]] = [
    ("Qwen/Qwen2.5-1.5B-Instruct",              False),
    ("Qwen/Qwen2.5-7B-Instruct",                False),
    ("Qwen/Qwen2.5-14B-Instruct",               False),
    ("meta-llama/Llama-3.1-8B-Instruct",        True),
    ("mistralai/Mistral-7B-Instruct-v0.3",      False),
    ("google/gemma-2-9b-it",                    True),
    ("meta-llama/Llama-3.1-70B-Instruct",       True),
    ("meta-llama/Llama-3.3-70B-Instruct",       True),
    ("meta-llama/Llama-3.2-3B-Instruct",        True),
    ("Qwen/Qwen2.5-32B-Instruct",               False),
    ("Qwen/Qwen2.5-72B-Instruct",               False),
    ("Qwen/Qwen2.5-Coder-7B-Instruct",          False),
    ("mistralai/Mistral-Small-24B-Instruct-2501", False),
    ("mistralai/Mistral-Nemo-Instruct-2407",    False),
    ("google/gemma-2-2b-it",                    True),
    ("google/gemma-2-27b-it",                   True),
    ("microsoft/Phi-3-mini-4k-instruct",        False),
    ("microsoft/Phi-3-small-8k-instruct",       False),
    ("microsoft/Phi-3-medium-4k-instruct",      False),
    ("deepseek-ai/DeepSeek-V2-Lite-Chat",       False),
    ("CohereForAI/c4ai-command-r-v01",          False),
    ("databricks/dbrx-instruct",                False),
]


@app.local_entrypoint()
def measure_batch(
    n_sequences: int = 100,
    max_length: int = 512,
    results_dir: str = "results",
    only_model: str = "",
) -> None:
    """
    Launch one Modal container per model and collect results locally.

    Parameters (pass as CLI flags)
    -------------------------------
    n_sequences   Number of calibration sequences  (default 100)
    max_length    Token length per sequence         (default 512)
    results_dir   Local directory to save results   (default "results/")
    only_model    If set, measure only this model ID

    Example
    -------
        modal run scripts/modal_run.py --n-sequences 50
        modal run scripts/modal_run.py --only-model "Qwen/Qwen2.5-7B-Instruct"
    """
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = MODELS
    if only_model:
        targets = [(m, t) for m, t in MODELS if m == only_model]
        if not targets:
            print(f"Model not found in registry: {only_model}")
            sys.exit(1)

    # Filter already-done models
    pending = []
    for model_name, _ in targets:
        slug = model_name.replace("/", "--").replace(" ", "_").lower()
        if not (out_dir / f"{slug}.json").exists():
            pending.append(model_name)
        else:
            print(f"  Skipping (already done): {model_name}")

    if not pending:
        print("All models already measured.")
        return

    print(f"\nMeasuring {len(pending)} models on Modal A100s …\n")

    # starmap launches all containers in parallel
    results = list(
        measure_model.starmap(
            [
                (model_name, n_sequences, max_length, 42, False)
                for model_name in pending
            ]
        )
    )

    # Save results locally
    succeeded = 0
    for model_name, result in zip(pending, results):
        if result is None:
            print(f"  ✗ No result returned for {model_name}")
            continue
        slug = model_name.replace("/", "--").replace(" ", "_").lower()
        out_path = out_dir / f"{slug}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        agg = result.get("aggregate", {})
        print(
            f"  ✓ {model_name}\n"
            f"      key_deff={agg.get('key_deff', '?'):.3f}  "
            f"val_deff={agg.get('val_deff', '?'):.3f}  "
            f"→ {out_path}"
        )
        succeeded += 1

    print(f"\n{succeeded}/{len(pending)} models measured successfully.")


# ---------------------------------------------------------------------------
# Single-model local entrypoint (convenient for quick tests)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def measure_single(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    n_sequences: int = 100,
    max_length: int = 512,
    output: str = "",
) -> None:
    """
    Measure a single model and print/save results.

    Example
    -------
        modal run scripts/modal_run.py::measure_single \\
            --model-name "meta-llama/Llama-3.1-8B-Instruct"
    """
    result = measure_model.remote(
        model_name=model_name,
        n_sequences=n_sequences,
        max_length=max_length,
    )

    agg = result["aggregate"]
    print(f"\n── Spectral Index ─────────────────────────────────────")
    print(f"  Model:     {model_name}")
    print(f"  key d_eff: {agg['key_deff']:.4f}  (κ={agg['key_kappa']:.4f})")
    print(f"  val d_eff: {agg['val_deff']:.4f}  (κ={agg['val_kappa']:.4f})")
    print(f"  dims95 (K/V): {agg['key_dims95']} / {agg['val_dims95']}")
    print(f"  dims99 (K/V): {agg['key_dims99']} / {agg['val_dims99']}")
    print("────────────────────────────────────────────────────────\n")

    out_path = output or f"results/{model_name.replace('/', '--').lower()}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {out_path}")
