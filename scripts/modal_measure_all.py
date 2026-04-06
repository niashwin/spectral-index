"""
The Spectral Index — Measure d_eff on all target models via Modal B200.
Runs each model sequentially on a single B200 GPU, saves results as JSON.

Usage:
    MODAL_PROFILE=sentra modal run scripts/modal_measure_all.py
"""

import modal
import json
import time

app = modal.App("spectral-index")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40",
        "datasets",
        "accelerate",
        "numpy",
        "sentencepiece",
        "protobuf",
    )
)

# All models to measure — (hf_name, needs_hf_token, short_name)
ALL_MODELS = [
    # Already verified in SpectralQuant paper — re-measure for fresh per-layer data
    ("Qwen/Qwen2.5-1.5B-Instruct", False, "qwen2.5-1.5b-instruct"),
    ("Qwen/Qwen2.5-7B-Instruct", False, "qwen2.5-7b-instruct"),
    ("Qwen/Qwen2.5-14B-Instruct", False, "qwen2.5-14b-instruct"),
    ("meta-llama/Llama-3.1-8B-Instruct", True, "llama-3.1-8b"),
    ("mistralai/Mistral-7B-Instruct-v0.3", False, "mistral-7b-v0.3"),
    ("google/gemma-2-9b-it", True, "gemma-2-9b-it"),
    # New models to measure
    ("meta-llama/Llama-3.2-3B-Instruct", True, "llama-3.2-3b"),
    ("meta-llama/Llama-3.1-70B-Instruct", True, "llama-3.1-70b"),
    ("meta-llama/Llama-3.3-70B-Instruct", True, "llama-3.3-70b"),
    ("Qwen/Qwen2.5-32B-Instruct", False, "qwen2.5-32b-instruct"),
    ("Qwen/Qwen2.5-72B-Instruct", False, "qwen2.5-72b-instruct"),
    ("Qwen/Qwen2.5-Coder-7B-Instruct", False, "qwen2.5-coder-7b"),
    ("mistralai/Mistral-Small-24B-Instruct-2501", False, "mistral-small-24b"),
    ("mistralai/Mistral-Nemo-Instruct-2407", False, "mistral-nemo-12b"),
    ("google/gemma-2-2b-it", True, "gemma-2-2b-it"),
    ("google/gemma-2-27b-it", True, "gemma-2-27b-it"),
    ("microsoft/Phi-3-mini-4k-instruct", False, "phi-3-mini"),
    ("microsoft/Phi-3-small-8k-instruct", False, "phi-3-small"),
    ("microsoft/Phi-3-medium-4k-instruct", False, "phi-3-medium"),
    ("deepseek-ai/DeepSeek-V2-Lite-Chat", False, "deepseek-v2-lite"),
    ("CohereForAI/c4ai-command-r-v01", True, "command-r-35b"),
    ("databricks/dbrx-instruct", True, "dbrx-132b"),
]

# Volume for persisting results across runs
vol = modal.Volume.from_name("spectral-index-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="B200",
    timeout=7200,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/results": vol},
)
def measure_model(model_name: str, needs_token: bool, short_name: str, n_sequences: int = 100):
    """Measure d_eff for a single model on B200 GPU."""
    import os
    import torch
    import numpy as np

    # Check if already done
    out_path = f"/results/{short_name}.json"
    if os.path.exists(out_path):
        print(f"[SKIP] {short_name} already measured")
        with open(out_path) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"Measuring: {model_name}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    t_start = time.time()

    # ── Load model ──
    token = os.environ.get("HF_TOKEN") if needs_token else None
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # For very large models, use auto device map
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=token,
        trust_remote_code=True,
    )
    model.eval()

    # ── Extract architecture info ──
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n_params = sum(p.numel() for p in model.parameters()) / 1e9

    # Detect actual KV head_dim from a test forward pass
    device = next(model.parameters()).device
    test_enc = tokenizer("test", return_tensors="pt").to(device)
    with torch.no_grad():
        test_out = model(**test_enc, use_cache=True)
    kv_test = test_out.past_key_values
    try:
        actual_hd = kv_test.key_cache[0].shape[-1]
        actual_n_kv = kv_test.key_cache[0].shape[1]
    except Exception:
        try:
            actual_hd = kv_test[0][0].shape[-1]
            actual_n_kv = kv_test[0][0].shape[1]
        except Exception:
            actual_hd = hd
            actual_n_kv = n_kv
    if actual_hd != hd:
        print(f"  KV head_dim from forward pass: {actual_hd} (config said {hd})")
        hd = actual_hd
    if actual_n_kv != n_kv:
        print(f"  KV n_heads from forward pass: {actual_n_kv} (config said {n_kv})")
        n_kv = actual_n_kv

    print(f"  Architecture: {n_layers} layers, {n_kv} KV heads, head_dim={hd}, {n_params:.1f}B params")

    # ── Load calibration data ──
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[:{n_sequences * 5}]")

    # ── Accumulate covariance matrices ──
    cov_keys = {
        (l, h): {"xtx": torch.zeros(hd, hd, dtype=torch.float64, device="cpu"), "n": 0}
        for l in range(n_layers) for h in range(n_kv)
    }
    cov_vals = {
        (l, h): {"xtx": torch.zeros(hd, hd, dtype=torch.float64, device="cpu"), "n": 0}
        for l in range(n_layers) for h in range(n_kv)
    }

    def extract_kv(kv, layer_idx):
        """Extract key/value tensors with multiple fallback strategies."""
        try:
            return kv.key_cache[layer_idx].float().cpu(), kv.value_cache[layer_idx].float().cpu()
        except Exception:
            pass
        try:
            return kv[layer_idx][0].float().cpu(), kv[layer_idx][1].float().cpu()
        except Exception:
            pass
        entry = list(kv)[layer_idx]
        return entry[0].float().cpu(), entry[1].float().cpu()

    n_done = 0
    t_calib = time.time()
    for item in ds:
        text = item.get("text", "")
        if len(text.strip()) < 100:
            continue
        enc = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
        if enc["input_ids"].shape[1] < 16:
            continue

        with torch.no_grad():
            out = model(**enc, use_cache=True)
            kv = out.past_key_values

        for l in range(n_layers):
            try:
                k_l, v_l = extract_kv(kv, l)
            except Exception as e:
                if n_done == 0:
                    print(f"  WARNING: Could not extract KV at layer {l}: {e}")
                continue

            for h in range(min(k_l.shape[1], n_kv)):
                X_key = k_l[0, h, :, :hd].double()
                cov_keys[(l, h)]["xtx"] += X_key.T @ X_key
                cov_keys[(l, h)]["n"] += X_key.shape[0]

                X_val = v_l[0, h, :, :hd].double()
                cov_vals[(l, h)]["xtx"] += X_val.T @ X_val
                cov_vals[(l, h)]["n"] += X_val.shape[0]

        n_done += 1
        if n_done >= n_sequences:
            break
        if n_done % 25 == 0:
            elapsed = time.time() - t_calib
            print(f"  Calibration: {n_done}/{n_sequences} sequences ({elapsed:.0f}s)")

    print(f"  Calibration done: {n_done} sequences in {time.time() - t_calib:.1f}s")

    # ── Eigendecompose ──
    def eigendecompose(cov_dict):
        results = {}
        for l in range(n_layers):
            for h in range(n_kv):
                n_samples = cov_dict[(l, h)]["n"]
                if n_samples < 2:
                    results[(l, h)] = {"d_eff": 0, "kappa": 0, "ev": [], "dims95": 0, "dims99": 0}
                    continue
                C = (cov_dict[(l, h)]["xtx"] / n_samples).float()
                ev, evec = torch.linalg.eigh(C)
                ev = ev.flip(0).clamp(min=0)

                # Participation ratio
                sum_ev = ev.sum()
                sum_ev_sq = (ev ** 2).sum()
                d_eff = float((sum_ev ** 2) / sum_ev_sq) if sum_ev_sq > 1e-12 else 1.0

                # Spectral gap
                d_eff_idx = max(1, min(round(d_eff), hd - 1))
                lam_k = float(ev[d_eff_idx - 1])
                lam_k1 = float(ev[min(d_eff_idx, hd - 1)])
                kappa = lam_k / max(lam_k1, 1e-10)
                kappa = min(kappa, 1e6)

                # Cumulative variance
                cumvar = ev.cumsum(0) / max(sum_ev, 1e-12)
                dims95 = int((cumvar < 0.95).sum().item()) + 1
                dims99 = int((cumvar < 0.99).sum().item()) + 1

                results[(l, h)] = {
                    "d_eff": d_eff,
                    "kappa": kappa,
                    "ev": ev[:min(64, hd)].tolist(),  # Save first 64 eigenvalues
                    "dims95": min(dims95, hd),
                    "dims99": min(dims99, hd),
                }
        return results

    print("  Computing eigenspectral statistics...")
    eigen_keys = eigendecompose(cov_keys)
    eigen_vals = eigendecompose(cov_vals)

    # ── Aggregate results ──
    all_key_deffs = [eigen_keys[(l, h)]["d_eff"] for l in range(n_layers) for h in range(n_kv) if eigen_keys[(l, h)]["d_eff"] > 0]
    all_val_deffs = [eigen_vals[(l, h)]["d_eff"] for l in range(n_layers) for h in range(n_kv) if eigen_vals[(l, h)]["d_eff"] > 0]
    all_kappas = [eigen_keys[(l, h)]["kappa"] for l in range(n_layers) for h in range(n_kv) if eigen_keys[(l, h)]["kappa"] > 0]

    mean_key_deff = float(np.mean(all_key_deffs)) if all_key_deffs else 0
    mean_val_deff = float(np.mean(all_val_deffs)) if all_val_deffs else 0
    mean_kappa = float(np.mean(all_kappas)) if all_kappas else 0

    # Per-layer means
    key_deff_per_layer = [
        float(np.mean([eigen_keys[(l, h)]["d_eff"] for h in range(n_kv)]))
        for l in range(n_layers)
    ]
    val_deff_per_layer = [
        float(np.mean([eigen_vals[(l, h)]["d_eff"] for h in range(n_kv)]))
        for l in range(n_layers)
    ]
    kappa_per_layer = [
        float(np.mean([eigen_keys[(l, h)]["kappa"] for h in range(n_kv)]))
        for l in range(n_layers)
    ]
    dims95_per_layer = [
        float(np.mean([eigen_keys[(l, h)]["dims95"] for h in range(n_kv)]))
        for l in range(n_layers)
    ]

    # Representative eigenvalue spectra (3 layers: early, mid, late)
    rep_layers = [0, n_layers // 2, n_layers - 1]
    rep_spectra = {}
    for l in rep_layers:
        rep_spectra[f"layer_{l}"] = {
            "key_eigenvalues": eigen_keys[(l, 0)]["ev"],
            "val_eigenvalues": eigen_vals[(l, 0)]["ev"],
            "key_d_eff": eigen_keys[(l, 0)]["d_eff"],
            "val_d_eff": eigen_vals[(l, 0)]["d_eff"],
        }

    total_time = time.time() - t_start

    result = {
        "model": model_name,
        "short_name": short_name,
        "n_layers": n_layers,
        "n_kv_heads": n_kv,
        "head_dim": hd,
        "n_params_b": round(n_params, 2),
        "n_sequences": n_done,
        "max_length": 512,
        "calibration_dataset": "wikitext-103-raw-v1",
        "measurement_time_s": round(total_time, 1),
        "gpu": torch.cuda.get_device_name(0),
        "key_deff": round(mean_key_deff, 4),
        "val_deff": round(mean_val_deff, 4),
        "kappa": round(mean_kappa, 4),
        "key_pct": round(mean_key_deff / hd * 100, 2),
        "key_deff_per_layer": [round(x, 4) for x in key_deff_per_layer],
        "val_deff_per_layer": [round(x, 4) for x in val_deff_per_layer],
        "kappa_per_layer": [round(x, 4) for x in kappa_per_layer],
        "dims95_per_layer": [round(x, 1) for x in dims95_per_layer],
        "representative_spectra": rep_spectra,
    }

    print(f"\n  RESULT: key_deff={mean_key_deff:.2f}, val_deff={mean_val_deff:.2f}, "
          f"kappa={mean_kappa:.3f}, key%={mean_key_deff/hd*100:.1f}%")
    print(f"  Total time: {total_time:.1f}s")

    # Save to volume
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print(f"  Saved: {out_path}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return result


@app.local_entrypoint()
def main():
    """Run d_eff measurement on all models."""
    print(f"Launching measurements for {len(ALL_MODELS)} models on B200...\n")

    results = []
    failed = []

    for model_name, needs_token, short_name in ALL_MODELS:
        print(f"\n>>> {short_name} ({model_name})")
        try:
            result = measure_model.remote(model_name, needs_token, short_name)
            results.append(result)
            print(f"  ✓ key_deff={result['key_deff']:.2f}, val_deff={result['val_deff']:.2f}")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed.append((short_name, str(e)))

    print(f"\n{'='*60}")
    print(f"DONE: {len(results)} succeeded, {len(failed)} failed")
    if failed:
        print("Failed models:")
        for name, err in failed:
            print(f"  - {name}: {err}")

    # Save all results locally too
    import os
    os.makedirs("results", exist_ok=True)
    for r in results:
        path = f"results/{r['short_name']}.json"
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        print(f"  Saved: {path}")

    print(f"\nAll results saved to results/")
