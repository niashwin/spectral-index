"""
measure.py — Spectral Index measurement script.

Runs N calibration sequences from WikiText-103 through a HuggingFace model,
collects KV cache vectors at every layer/head, builds uncentered covariance
matrices, and computes:

  * d_eff  (participation ratio): (Σλ_i)² / Σ(λ_i²)
  * kappa  (spectral gap):        λ_{round(d_eff)} / λ_{round(d_eff)+1}
  * dims95 / dims99               cumulative-variance thresholds

Results are written as JSON with full provenance so measurements are
reproducible and citable.

Usage
-----
    python -m src.measure \\
        --model "meta-llama/Llama-3.1-8B-Instruct" \\
        --n-sequences 100 \\
        --max-length 512 \\
        --output results/llama-3.1-8b.json \\
        --device cuda

Environment variables
---------------------
    HF_TOKEN   HuggingFace token for gated models (e.g. Llama, Gemma)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def _device_for(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def _model_dtype(device: torch.device) -> torch.dtype:
    """Pick a sensible inference dtype based on hardware."""
    if device.type == "cuda":
        return torch.float16
    return torch.float32  # MPS and CPU are safer with fp32


# ---------------------------------------------------------------------------
# Model / tokenizer loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    hf_token: str | None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer from HuggingFace Hub."""
    tok_kwargs: dict[str, Any] = {"use_fast": True}
    if hf_token:
        tok_kwargs["token"] = hf_token

    logger.info("Loading tokenizer for %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": "auto" if device.type == "cuda" else None,
        "low_cpu_mem_usage": True,
    }
    if hf_token:
        model_kwargs["token"] = hf_token

    logger.info("Loading model %s (dtype=%s, device=%s)", model_name, dtype, device)
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if device.type != "cuda":
        model = model.to(device)

    model.eval()
    logger.info("Model loaded — %s parameters", f"{sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Architecture introspection
# ---------------------------------------------------------------------------

def _get_arch_params(model: AutoModelForCausalLM) -> tuple[int, int, int]:
    """
    Return (n_layers, n_kv_heads, head_dim).

    Handles standard Transformers config fields and GQA (num_key_value_heads).
    """
    cfg = model.config

    # --- number of layers ---
    n_layers = (
        getattr(cfg, "num_hidden_layers", None)
        or getattr(cfg, "n_layer", None)
        or getattr(cfg, "num_layers", None)
    )
    if n_layers is None:
        raise ValueError("Cannot determine number of layers from config")

    # --- number of KV heads (GQA-aware) ---
    n_kv_heads = (
        getattr(cfg, "num_key_value_heads", None)
        or getattr(cfg, "n_head_kv", None)
        or getattr(cfg, "num_attention_heads", None)
        or getattr(cfg, "n_head", None)
    )
    if n_kv_heads is None:
        raise ValueError("Cannot determine n_kv_heads from config")

    # --- head dimension ---
    # Try direct config attribute first, then derive from hidden size.
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        n_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
        hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
        if n_heads and hidden:
            head_dim = hidden // n_heads
        else:
            raise ValueError("Cannot determine head_dim from config")

    logger.info(
        "Architecture: %d layers, %d KV heads, head_dim=%d",
        n_layers, n_kv_heads, head_dim,
    )
    return int(n_layers), int(n_kv_heads), int(head_dim)


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

def _get_calibration_sequences(
    tokenizer: AutoTokenizer,
    n_sequences: int,
    max_length: int,
    seed: int = 42,
) -> list[dict]:
    """
    Draw n_sequences non-overlapping windows from WikiText-103 validation split.
    Returns a list of tokenizer output dicts (input_ids, attention_mask).
    """
    logger.info("Loading WikiText-103 calibration data (n=%d, max_len=%d)", n_sequences, max_length)
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation", trust_remote_code=True)

    # Concatenate all text, then chunk into fixed-length windows.
    full_text = "\n\n".join(t for t in dataset["text"] if t.strip())
    all_ids = tokenizer.encode(full_text, add_special_tokens=False)

    sequences = []
    torch.manual_seed(seed)
    total_windows = len(all_ids) // max_length
    if total_windows < n_sequences:
        logger.warning(
            "Only %d non-overlapping windows available; requested %d. "
            "Reducing to %d.",
            total_windows, n_sequences, total_windows,
        )
        n_sequences = total_windows

    # Use evenly-spaced windows for coverage rather than random sampling
    # so results are deterministic and span the whole corpus.
    indices = torch.linspace(0, total_windows - 1, n_sequences).long().tolist()
    for idx in indices:
        start = idx * max_length
        chunk = all_ids[start : start + max_length]
        ids = torch.tensor(chunk, dtype=torch.long).unsqueeze(0)
        attn = torch.ones_like(ids)
        sequences.append({"input_ids": ids, "attention_mask": attn})

    logger.info("Prepared %d calibration sequences of length %d", len(sequences), max_length)
    return sequences


# ---------------------------------------------------------------------------
# KV cache extraction (multi-pattern with fallbacks)
# ---------------------------------------------------------------------------

def _extract_kv(past_key_values: Any, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract (k, v) tensors for *layer* from past_key_values.

    Handles:
      1. transformers.cache_utils.DynamicCache   (.key_cache / .value_cache lists)
      2. Tuple-of-tuples                          past_key_values[l][0/1]
      3. Any other iterable                       list(past_key_values)[l][0/1]

    Shape returned: (batch, n_kv_heads, seq_len, head_dim)
    """
    # Pattern 1: DynamicCache (transformers ≥ 4.36)
    try:
        k = past_key_values.key_cache[layer].float().cpu()
        v = past_key_values.value_cache[layer].float().cpu()
        return k, v
    except (AttributeError, IndexError, TypeError):
        pass

    # Pattern 2: Tuple indexing
    try:
        k = past_key_values[layer][0].float().cpu()
        v = past_key_values[layer][1].float().cpu()
        return k, v
    except (TypeError, IndexError):
        pass

    # Pattern 3: Generic iterable
    entry = list(past_key_values)[layer]
    k = entry[0].float().cpu()
    v = entry[1].float().cpu()
    return k, v


# ---------------------------------------------------------------------------
# Spectral metrics
# ---------------------------------------------------------------------------

def _compute_spectral_metrics(
    cov_dict: dict[str, dict],
    head_dim: int,
) -> tuple[float, float, int, int, list[float]]:
    """
    Given {"xtx": Tensor[hd,hd], "n": int} accumulate over all heads in
    cov_dict, then compute aggregate spectral metrics.

    Returns
    -------
    d_eff, kappa, dims95, dims99, eigenvalues (descending, first 32 or hd)
    """
    # Build pooled covariance (average across all heads)
    xtx_pool = torch.zeros(head_dim, head_dim, dtype=torch.float64)
    n_total = 0
    for entry in cov_dict.values():
        xtx_pool += entry["xtx"]
        n_total += entry["n"]
    if n_total == 0:
        raise ValueError("No samples collected")

    C = (xtx_pool / n_total).float()
    ev, _ = torch.linalg.eigh(C)
    ev = ev.flip(0).clamp(min=0.0)  # descending, non-negative

    d_eff = _participation_ratio(ev)
    kappa = _spectral_gap(ev, d_eff)
    dims95 = _dims_for_variance(ev, 0.95)
    dims99 = _dims_for_variance(ev, 0.99)
    spectra = ev[:32].tolist()  # representative prefix

    return d_eff, kappa, dims95, dims99, spectra


def _participation_ratio(ev: torch.Tensor) -> float:
    """d_eff = (Σλ_i)² / Σ(λ_i²)"""
    s1 = ev.sum()
    s2 = (ev ** 2).sum()
    if s2 == 0:
        return float(len(ev))
    return (s1 ** 2 / s2).item()


def _spectral_gap(ev: torch.Tensor, d_eff: float) -> float:
    """
    kappa = λ_{round(d_eff)} / λ_{round(d_eff)+1}

    Indices are 0-based.  round(d_eff) corresponds to the last eigenvalue
    "inside" the effective subspace.
    """
    idx = round(d_eff) - 1  # 0-based index of the d_eff-th eigenvalue
    idx = max(0, min(idx, len(ev) - 2))
    denom = ev[idx + 1]
    if denom == 0:
        return float("inf")
    return (ev[idx] / denom).item()


def _dims_for_variance(ev: torch.Tensor, threshold: float) -> int:
    """Minimum number of eigendimensions to explain `threshold` fraction of variance."""
    total = ev.sum()
    if total == 0:
        return len(ev)
    cumvar = ev.cumsum(0) / total
    above = (cumvar >= threshold).nonzero(as_tuple=True)[0]
    if len(above) == 0:
        return len(ev)
    return (above[0].item() + 1)


def _per_layer_metrics(
    cov_keys_layer: dict[int, dict],
    cov_vals_layer: dict[int, dict],
    head_dim: int,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Compute per-layer mean d_eff and mean kappa for keys and values.

    cov_keys_layer: {head_idx: {"xtx": ..., "n": ...}}
    Returns: key_deff_per_layer, key_kappa_per_layer,
             val_deff_per_layer, val_kappa_per_layer
    """
    heads = sorted(cov_keys_layer.keys())
    key_deffs, key_kappas = [], []
    val_deffs, val_kappas = [], []

    for h in heads:
        # Keys
        n_k = cov_keys_layer[h]["n"]
        if n_k > 0:
            Ck = (cov_keys_layer[h]["xtx"] / n_k).float()
            ev_k, _ = torch.linalg.eigh(Ck)
            ev_k = ev_k.flip(0).clamp(min=0.0)
            key_deffs.append(_participation_ratio(ev_k))
            key_kappas.append(_spectral_gap(ev_k, key_deffs[-1]))

        # Values
        n_v = cov_vals_layer[h]["n"]
        if n_v > 0:
            Cv = (cov_vals_layer[h]["xtx"] / n_v).float()
            ev_v, _ = torch.linalg.eigh(Cv)
            ev_v = ev_v.flip(0).clamp(min=0.0)
            val_deffs.append(_participation_ratio(ev_v))
            val_kappas.append(_spectral_gap(ev_v, val_deffs[-1]))

    def mean(lst: list) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    return (
        [round(x, 4) for x in key_deffs],
        [round(x, 4) for x in key_kappas],
        [round(x, 4) for x in val_deffs],
        [round(x, 4) for x in val_kappas],
    )


# ---------------------------------------------------------------------------
# Main measurement routine
# ---------------------------------------------------------------------------

def measure(
    model_name: str,
    n_sequences: int = 100,
    max_length: int = 512,
    output_path: Path | None = None,
    device_str: str = "auto",
    hf_token: str | None = None,
    seed: int = 42,
) -> dict:
    """
    Full measurement pipeline. Returns the result dict.

    This function is designed to be importable and callable from external
    scripts (e.g. Modal runners) as well as the CLI.
    """
    device = _device_for(device_str)
    dtype = _model_dtype(device)

    # --- Load model ---
    model, tokenizer = load_model_and_tokenizer(model_name, device, dtype, hf_token)
    n_layers, n_kv, head_dim = _get_arch_params(model)

    # --- Calibration data ---
    sequences = _get_calibration_sequences(tokenizer, n_sequences, max_length, seed=seed)
    actual_n = len(sequences)

    # --- Accumulator initialisation ---
    # Keyed by (layer, head) → {"xtx": Tensor[hd,hd], "n": int}
    cov_keys: dict[tuple[int, int], dict] = {
        (l, h): {"xtx": torch.zeros(head_dim, head_dim, dtype=torch.float64), "n": 0}
        for l in range(n_layers)
        for h in range(n_kv)
    }
    cov_vals: dict[tuple[int, int], dict] = {
        (l, h): {"xtx": torch.zeros(head_dim, head_dim, dtype=torch.float64), "n": 0}
        for l in range(n_layers)
        for h in range(n_kv)
    }

    # --- Crash-safe partial-save setup ---
    result: dict = {}
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    def _partial_save(sig=None, frame=None):
        if output_path and result:
            partial_path = output_path.with_suffix(".partial.json")
            with open(partial_path, "w") as f:
                json.dump(result, f, indent=2)
            logger.info("Partial results saved to %s", partial_path)
        if sig is not None:
            sys.exit(1)

    signal.signal(signal.SIGINT, _partial_save)
    signal.signal(signal.SIGTERM, _partial_save)

    # --- Forward passes ---
    t_start = time.time()
    with torch.no_grad():
        for seq_idx, batch in enumerate(sequences):
            if (seq_idx + 1) % 10 == 0 or seq_idx == 0:
                elapsed = time.time() - t_start
                eta = (elapsed / (seq_idx + 1)) * (actual_n - seq_idx - 1)
                logger.info(
                    "Sequence %d/%d  (elapsed %.0fs, ETA %.0fs)",
                    seq_idx + 1, actual_n, elapsed, eta,
                )

            # Move inputs to device
            enc = {k: v.to(device) for k, v in batch.items()}

            try:
                out = model(**enc, use_cache=True)
            except RuntimeError as e:
                logger.warning("Forward pass failed for sequence %d: %s — skipping", seq_idx, e)
                continue

            past = out.past_key_values

            for l in range(n_layers):
                try:
                    k_l, v_l = _extract_kv(past, l)
                except Exception as e:
                    logger.error("KV extraction failed at layer %d: %s", l, e)
                    continue

                # k_l / v_l shape: (batch=1, n_kv_heads, seq_len, head_dim)
                # Some models expand KV heads to match Q heads; re-deduplicate.
                actual_kv = k_l.shape[1]
                if actual_kv != n_kv:
                    # Take every (actual_kv // n_kv)-th head
                    stride = actual_kv // n_kv
                    k_l = k_l[:, ::stride, :, :][:, :n_kv, :, :]
                    v_l = v_l[:, ::stride, :, :][:, :n_kv, :, :]

                for h in range(n_kv):
                    # X_key: (seq_len, head_dim)
                    X_k = k_l[0, h, :, :].double()  # float64 for numerical stability
                    X_v = v_l[0, h, :, :].double()

                    cov_keys[(l, h)]["xtx"] += X_k.T @ X_k
                    cov_keys[(l, h)]["n"] += X_k.shape[0]
                    cov_vals[(l, h)]["xtx"] += X_v.T @ X_v
                    cov_vals[(l, h)]["n"] += X_v.shape[0]

            # Free GPU memory
            del out, past
            if device.type == "cuda":
                torch.cuda.empty_cache()

    t_elapsed = time.time() - t_start
    logger.info("Forward passes complete in %.1fs", t_elapsed)

    # --- Aggregate spectral metrics ---
    logger.info("Computing spectral metrics …")

    key_deff, key_kappa, key_dims95, key_dims99, key_spectra = _compute_spectral_metrics(
        cov_keys, head_dim
    )
    val_deff, val_kappa, val_dims95, val_dims99, val_spectra = _compute_spectral_metrics(
        cov_vals, head_dim
    )

    # --- Per-layer breakdown ---
    key_deff_layers: list[float] = []
    key_kappa_layers: list[float] = []
    val_deff_layers: list[float] = []
    val_kappa_layers: list[float] = []

    for l in range(n_layers):
        layer_cov_k = {h: cov_keys[(l, h)] for h in range(n_kv)}
        layer_cov_v = {h: cov_vals[(l, h)] for h in range(n_kv)}
        kd, kk, vd, vk = _per_layer_metrics(layer_cov_k, layer_cov_v, head_dim)
        key_deff_layers.append(round(sum(kd) / len(kd), 4) if kd else 0.0)
        key_kappa_layers.append(round(sum(kk) / len(kk), 4) if kk else 0.0)
        val_deff_layers.append(round(sum(vd) / len(vd), 4) if vd else 0.0)
        val_kappa_layers.append(round(sum(vk) / len(vk), 4) if vk else 0.0)

    # --- Assemble result ---
    result = {
        "schema_version": "1.0",
        "provenance": {
            "model_name": model_name,
            "n_sequences": actual_n,
            "max_length": max_length,
            "seed": seed,
            "measured_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(t_elapsed, 1),
            "transformers_version": _pkg_version("transformers"),
            "torch_version": torch.__version__,
        },
        "architecture": {
            "n_layers": n_layers,
            "n_kv_heads": n_kv,
            "head_dim": head_dim,
        },
        "aggregate": {
            "key_deff": round(key_deff, 4),
            "key_kappa": round(key_kappa, 4),
            "key_dims95": key_dims95,
            "key_dims99": key_dims99,
            "val_deff": round(val_deff, 4),
            "val_kappa": round(val_kappa, 4),
            "val_dims95": val_dims95,
            "val_dims99": val_dims99,
        },
        "per_layer": {
            "key_deff": key_deff_layers,
            "key_kappa": key_kappa_layers,
            "val_deff": val_deff_layers,
            "val_kappa": val_kappa_layers,
        },
        "representative_spectra": {
            "key_eigenvalues": [round(x, 6) for x in key_spectra],
            "val_eigenvalues": [round(x, 6) for x in val_spectra],
        },
    }

    # --- Save ---
    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Results written to %s", output_path)

    return result


def _pkg_version(pkg: str) -> str:
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.measure",
        description="Measure effective KV-cache dimensionality of a HuggingFace LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        metavar="HF_MODEL_ID",
        help="HuggingFace model name or local path, e.g. meta-llama/Llama-3.1-8B-Instruct",
    )
    p.add_argument(
        "--n-sequences",
        type=int,
        default=100,
        metavar="N",
        help="Number of WikiText-103 calibration sequences",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=512,
        metavar="L",
        help="Token length of each calibration sequence",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output JSON file path (default: results/<model_slug>.json)",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Compute device",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return p


def _default_output(model_name: str) -> Path:
    slug = model_name.replace("/", "--").replace(" ", "_").lower()
    return Path("results") / f"{slug}.json"


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.debug("HF_TOKEN not set — only public models accessible")

    output = args.output or _default_output(args.model)

    result = measure(
        model_name=args.model,
        n_sequences=args.n_sequences,
        max_length=args.max_length,
        output_path=output,
        device_str=args.device,
        hf_token=hf_token,
        seed=args.seed,
    )

    # Print summary to stdout
    agg = result["aggregate"]
    print("\n── Spectral Index Results ──────────────────────────────")
    print(f"  Model:       {args.model}")
    print(f"  key d_eff:   {agg['key_deff']:.3f}   (dims95={agg['key_dims95']}, dims99={agg['key_dims99']})")
    print(f"  key κ:       {agg['key_kappa']:.3f}")
    print(f"  val d_eff:   {agg['val_deff']:.3f}   (dims95={agg['val_dims95']}, dims99={agg['val_dims99']})")
    print(f"  val κ:       {agg['val_kappa']:.3f}")
    print(f"  Output:      {output}")
    print("────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
