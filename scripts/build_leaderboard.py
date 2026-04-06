"""
build_leaderboard.py — Aggregate per-model JSON results into site/data.json.

Reads every *.json file in results/ (excluding *.partial.json), validates
the schema, and writes a single site/data.json that the leaderboard website
consumes.

Usage
-----
    python scripts/build_leaderboard.py [--results-dir results/] [--out site/data.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known metadata for labelling
# ---------------------------------------------------------------------------

# (model_id_prefix_or_exact) → friendly display name
DISPLAY_NAMES: dict[str, str] = {
    "meta-llama/Llama-3.1-8B-Instruct":            "Llama 3.1 8B Instruct",
    "meta-llama/Llama-3.1-70B-Instruct":           "Llama 3.1 70B Instruct",
    "meta-llama/Llama-3.3-70B-Instruct":           "Llama 3.3 70B Instruct",
    "meta-llama/Llama-3.2-3B-Instruct":            "Llama 3.2 3B Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct":                  "Qwen 2.5 1.5B Instruct",
    "Qwen/Qwen2.5-7B-Instruct":                    "Qwen 2.5 7B Instruct",
    "Qwen/Qwen2.5-14B-Instruct":                   "Qwen 2.5 14B Instruct",
    "Qwen/Qwen2.5-32B-Instruct":                   "Qwen 2.5 32B Instruct",
    "Qwen/Qwen2.5-72B-Instruct":                   "Qwen 2.5 72B Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct":              "Qwen 2.5 Coder 7B",
    "mistralai/Mistral-7B-Instruct-v0.3":           "Mistral 7B v0.3",
    "mistralai/Mistral-Small-24B-Instruct-2501":    "Mistral Small 24B",
    "mistralai/Mistral-Nemo-Instruct-2407":         "Mistral Nemo 12B",
    "google/gemma-2-2b-it":                         "Gemma 2 2B",
    "google/gemma-2-9b-it":                         "Gemma 2 9B",
    "google/gemma-2-27b-it":                        "Gemma 2 27B",
    "microsoft/Phi-3-mini-4k-instruct":             "Phi-3 Mini",
    "microsoft/Phi-3-small-8k-instruct":            "Phi-3 Small",
    "microsoft/Phi-3-medium-4k-instruct":           "Phi-3 Medium",
    "deepseek-ai/DeepSeek-V2-Lite-Chat":            "DeepSeek-V2 Lite",
    "CohereForAI/c4ai-command-r-v01":               "Command-R",
    "databricks/dbrx-instruct":                     "DBRX Instruct",
}

FAMILIES: dict[str, str] = {
    "meta-llama": "Llama",
    "Qwen": "Qwen",
    "mistralai": "Mistral",
    "google": "Gemma",
    "microsoft": "Phi",
    "deepseek-ai": "DeepSeek",
    "CohereForAI": "Cohere",
    "databricks": "DBRX",
}

# Approximate parameter counts (billions) for display
PARAM_COUNTS: dict[str, float] = {
    "meta-llama/Llama-3.1-8B-Instruct":           8.0,
    "meta-llama/Llama-3.1-70B-Instruct":         70.0,
    "meta-llama/Llama-3.3-70B-Instruct":         70.0,
    "meta-llama/Llama-3.2-3B-Instruct":           3.0,
    "Qwen/Qwen2.5-1.5B-Instruct":                 1.5,
    "Qwen/Qwen2.5-7B-Instruct":                   7.0,
    "Qwen/Qwen2.5-14B-Instruct":                 14.0,
    "Qwen/Qwen2.5-32B-Instruct":                 32.0,
    "Qwen/Qwen2.5-72B-Instruct":                 72.0,
    "Qwen/Qwen2.5-Coder-7B-Instruct":             7.0,
    "mistralai/Mistral-7B-Instruct-v0.3":          7.0,
    "mistralai/Mistral-Small-24B-Instruct-2501":  24.0,
    "mistralai/Mistral-Nemo-Instruct-2407":       12.0,
    "google/gemma-2-2b-it":                        2.0,
    "google/gemma-2-9b-it":                        9.0,
    "google/gemma-2-27b-it":                      27.0,
    "microsoft/Phi-3-mini-4k-instruct":            3.8,
    "microsoft/Phi-3-small-8k-instruct":           7.0,
    "microsoft/Phi-3-medium-4k-instruct":         14.0,
    "deepseek-ai/DeepSeek-V2-Lite-Chat":          16.0,
    "CohereForAI/c4ai-command-r-v01":             35.0,
    "databricks/dbrx-instruct":                  132.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _display_name(model_id: str) -> str:
    return DISPLAY_NAMES.get(model_id, model_id.split("/")[-1])


def _family(model_id: str) -> str:
    org = model_id.split("/")[0]
    return FAMILIES.get(org, org)


def _param_count(model_id: str) -> float | None:
    return PARAM_COUNTS.get(model_id)


def _load_result(path: Path) -> dict | None:
    try:
        with open(path) as f:
            data = json.load(f)
        # Minimal schema validation
        if "aggregate" not in data:
            logger.warning("Skipping %s: missing 'aggregate' key", path)
            return None
        return data
    except Exception as e:
        logger.warning("Skipping %s: %s", path, e)
        return None


def _build_row(result: dict) -> dict:
    """Convert a raw result dict into a leaderboard row."""
    prov = result.get("provenance", {})
    arch = result.get("architecture", {})
    agg  = result["aggregate"]
    per  = result.get("per_layer", {})

    model_id = prov.get("model_name", "unknown")

    row: dict = {
        # Identity
        "model_id":    model_id,
        "model_name":  _display_name(model_id),
        "family":      _family(model_id),
        "params_b":    _param_count(model_id),
        "hf_url":      f"https://huggingface.co/{model_id}",

        # Architecture
        "n_layers":    arch.get("n_layers"),
        "n_kv_heads":  arch.get("n_kv_heads"),
        "head_dim":    arch.get("head_dim"),

        # Aggregate spectral metrics
        "key_deff":    agg.get("key_deff"),
        "key_kappa":   agg.get("key_kappa"),
        "key_dims95":  agg.get("key_dims95"),
        "key_dims99":  agg.get("key_dims99"),
        "val_deff":    agg.get("val_deff"),
        "val_kappa":   agg.get("val_kappa"),
        "val_dims95":  agg.get("val_dims95"),
        "val_dims99":  agg.get("val_dims99"),

        # Per-layer data (included if present)
        "per_layer_key_deff":  per.get("key_deff"),
        "per_layer_key_kappa": per.get("key_kappa"),
        "per_layer_val_deff":  per.get("val_deff"),
        "per_layer_val_kappa": per.get("val_kappa"),

        # Representative eigenvalue spectra
        "key_eigenvalues": result.get("representative_spectra", {}).get("key_eigenvalues"),
        "val_eigenvalues": result.get("representative_spectra", {}).get("val_eigenvalues"),

        # Provenance / measurement metadata
        "n_sequences":   prov.get("n_sequences"),
        "max_length":    prov.get("max_length"),
        "measured_at":   prov.get("measured_at"),
        "schema_version": result.get("schema_version", "1.0"),
    }
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build(results_dir: Path, out_path: Path, verbose: bool = False) -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
    )

    json_files = sorted(
        p for p in results_dir.glob("*.json")
        if not p.name.endswith(".partial.json")
    )
    if not json_files:
        logger.error("No result files found in %s", results_dir)
        sys.exit(1)

    logger.info("Found %d result file(s) in %s", len(json_files), results_dir)

    rows: list[dict] = []
    for path in json_files:
        result = _load_result(path)
        if result is None:
            continue
        row = _build_row(result)
        rows.append(row)
        logger.info(
            "  %-45s  key_deff=%.3f  val_deff=%.3f",
            row["model_id"], row["key_deff"] or 0, row["val_deff"] or 0,
        )

    # Sort by key_deff ascending (most compressible first)
    rows.sort(key=lambda r: (r.get("key_deff") or 999))

    site_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_models": len(rows),
        "models": rows,
        "meta": {
            "description": (
                "Spectral Index leaderboard — effective dimensionality of KV cache "
                "representations across open-source LLMs, measured on WikiText-103."
            ),
            "paper": "https://arxiv.org/abs/2405.20582",
            "methodology": (
                "Uncentered covariance C = XᵀX/n, participation ratio d_eff = (Σλ)²/Σ(λ²), "
                "spectral gap κ = λ_{d_eff}/λ_{d_eff+1}, 100 sequences × 512 tokens."
            ),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(site_data, f, indent=2)

    logger.info("Leaderboard written to %s  (%d models)", out_path, len(rows))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python scripts/build_leaderboard.py",
        description="Aggregate measurement JSON files into site/data.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results-dir", type=Path, default=Path("results"),
        help="Directory containing per-model result JSON files",
    )
    p.add_argument(
        "--out", type=Path, default=Path("site/data.json"),
        help="Output path for the aggregated leaderboard JSON",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logging",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    build(results_dir=args.results_dir, out_path=args.out, verbose=args.verbose)
