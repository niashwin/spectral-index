"""
run_all.py — Batch runner for Spectral Index measurements.

Iterates over all target models, skipping any that already have a result
file, and calls src.measure.measure() for each one sequentially.

Usage
-----
    python scripts/run_all.py [--n-sequences 100] [--max-length 512] \\
                               [--results-dir results/] [--device cuda]

The script can be safely interrupted and restarted; completed models are
detected automatically via the presence of their output JSON files.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure the repo root is on the path when run directly.
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.measure import measure, _setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Target model registry
# (model_id, needs_hf_token)
# ---------------------------------------------------------------------------
MODELS: list[tuple[str, bool]] = [
    # ── Verified in SpectralQuant paper ─────────────────────────────────
    ("Qwen/Qwen2.5-1.5B-Instruct",              False),
    ("Qwen/Qwen2.5-7B-Instruct",                False),
    ("Qwen/Qwen2.5-14B-Instruct",               False),
    ("meta-llama/Llama-3.1-8B-Instruct",        True),
    ("mistralai/Mistral-7B-Instruct-v0.3",      False),
    ("google/gemma-2-9b-it",                    True),
    # ── New measurements ────────────────────────────────────────────────
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output_path(model_name: str, results_dir: Path) -> Path:
    slug = model_name.replace("/", "--").replace(" ", "_").lower()
    return results_dir / f"{slug}.json"


def _model_done(model_name: str, results_dir: Path) -> bool:
    return _output_path(model_name, results_dir).exists()


def _run_model(
    model_name: str,
    needs_token: bool,
    results_dir: Path,
    n_sequences: int,
    max_length: int,
    device: str,
    seed: int,
) -> bool:
    """Run measurement for a single model. Returns True on success."""
    hf_token = os.environ.get("HF_TOKEN") if needs_token else os.environ.get("HF_TOKEN")

    output = _output_path(model_name, results_dir)
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("Measuring: %s", model_name)
    logger.info("Output:    %s", output)

    t0 = time.time()
    try:
        measure(
            model_name=model_name,
            n_sequences=n_sequences,
            max_length=max_length,
            output_path=output,
            device_str=device,
            hf_token=hf_token,
            seed=seed,
        )
        elapsed = time.time() - t0
        logger.info("✓ Done in %.0fs: %s", elapsed, model_name)
        return True

    except Exception as exc:
        elapsed = time.time() - t0
        logger.error("✗ Failed after %.0fs: %s\n  %s", elapsed, model_name, exc)
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python scripts/run_all.py",
        description="Run Spectral Index measurements for all target models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--n-sequences", type=int, default=100,
        help="Number of calibration sequences per model",
    )
    p.add_argument(
        "--max-length", type=int, default=512,
        help="Token length of each calibration sequence",
    )
    p.add_argument(
        "--results-dir", type=Path, default=Path("results"),
        help="Directory to read/write result JSON files",
    )
    p.add_argument(
        "--device", default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Compute device",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    p.add_argument(
        "--skip-done", action="store_true", default=True,
        help="Skip models that already have result files (default: on)",
    )
    p.add_argument(
        "--no-skip-done", dest="skip_done", action="store_false",
        help="Re-measure models even if result files exist",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logging",
    )
    # Allow targeting a specific model subset by index (1-based) e.g. --only 3 5 7
    p.add_argument(
        "--only", nargs="+", type=int, metavar="IDX",
        help="Only run models at these 1-based indices (from the MODELS list)",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)

    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Select models to run
    candidates = list(enumerate(MODELS, start=1))
    if args.only:
        candidates = [(i, m) for i, m in candidates if i in args.only]

    total = len(candidates)
    skipped = 0
    succeeded = 0
    failed: list[str] = []

    for idx, (model_name, needs_token) in candidates:
        if args.skip_done and _model_done(model_name, args.results_dir):
            logger.info("[%d/%d] Skipping (already done): %s", idx, len(MODELS), model_name)
            skipped += 1
            continue

        ok = _run_model(
            model_name=model_name,
            needs_token=needs_token,
            results_dir=args.results_dir,
            n_sequences=args.n_sequences,
            max_length=args.max_length,
            device=args.device,
            seed=args.seed,
        )
        if ok:
            succeeded += 1
        else:
            failed.append(model_name)

    # Summary
    logger.info("")
    logger.info("━━━━━━━ Run-all complete ━━━━━━━")
    logger.info("  Total:     %d", total)
    logger.info("  Skipped:   %d", skipped)
    logger.info("  Succeeded: %d", succeeded)
    logger.info("  Failed:    %d", len(failed))
    if failed:
        logger.info("  Failed models:")
        for m in failed:
            logger.info("    - %s", m)

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
