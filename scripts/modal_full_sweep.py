"""
The Spectral Index — Full parallel sweep on B200.

1. Measure d_eff for all remaining models (n=100 sequences)
2. Run calibration stability sweep (n=100, 1000, 10000) on 3 reference models
   to show d_eff saturates asymptotically.

Usage:
    MODAL_PROFILE=sentra modal run scripts/modal_full_sweep.py
"""

import modal
import json
import time

app = modal.App("spectral-index-sweep")

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

vol = modal.Volume.from_name("spectral-index-results", create_if_missing=True)

# ── Models that still need n=100 measurement ──
NEW_MODELS = [
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

# ── Reference models for calibration stability sweep ──
STABILITY_MODELS = [
    ("Qwen/Qwen2.5-7B-Instruct", False, "qwen2.5-7b-instruct"),
    ("meta-llama/Llama-3.1-8B-Instruct", True, "llama-3.1-8b"),
    ("mistralai/Mistral-7B-Instruct-v0.3", False, "mistral-7b-v0.3"),
]
STABILITY_N_VALUES = [100, 1000, 10000]


def _measure_core(model_name, needs_token, n_sequences, max_length=512):
    """Core measurement logic. Returns result dict."""
    import os
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    t_start = time.time()
    token = os.environ.get("HF_TOKEN") if needs_token else None

    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto",
        token=token, trust_remote_code=True,
    )
    model.eval()

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n_params = sum(p.numel() for p in model.parameters()) / 1e9

    # Detect actual dims from forward pass
    device = next(model.parameters()).device
    test_enc = tokenizer("test", return_tensors="pt").to(device)
    with torch.no_grad():
        test_out = model(**test_enc, use_cache=True)
    kv_test = test_out.past_key_values
    try:
        hd = kv_test.key_cache[0].shape[-1]
        n_kv = kv_test.key_cache[0].shape[1]
    except Exception:
        try:
            hd = kv_test[0][0].shape[-1]
            n_kv = kv_test[0][0].shape[1]
        except Exception:
            pass

    print(f"  Arch: {n_layers}L, {n_kv}KV, hd={hd}, {n_params:.1f}B params")

    # Calibration data
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[:{n_sequences * 5}]")

    cov_keys = {(l, h): {"xtx": torch.zeros(hd, hd, dtype=torch.float64, device="cpu"), "n": 0}
                for l in range(n_layers) for h in range(n_kv)}
    cov_vals = {(l, h): {"xtx": torch.zeros(hd, hd, dtype=torch.float64, device="cpu"), "n": 0}
                for l in range(n_layers) for h in range(n_kv)}

    def extract_kv(kv, li):
        try: return kv.key_cache[li].float().cpu(), kv.value_cache[li].float().cpu()
        except Exception: pass
        try: return kv[li][0].float().cpu(), kv[li][1].float().cpu()
        except Exception: pass
        e = list(kv)[li]
        return e[0].float().cpu(), e[1].float().cpu()

    n_done = 0
    t_cal = time.time()
    for item in ds:
        text = item.get("text", "")
        if len(text.strip()) < 100:
            continue
        enc = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True).to(device)
        if enc["input_ids"].shape[1] < 16:
            continue
        with torch.no_grad():
            out = model(**enc, use_cache=True)
            kv = out.past_key_values
        for l in range(n_layers):
            try:
                k_l, v_l = extract_kv(kv, l)
            except Exception:
                continue
            for h in range(min(k_l.shape[1], n_kv)):
                X_k = k_l[0, h, :, :hd].double()
                cov_keys[(l, h)]["xtx"] += X_k.T @ X_k
                cov_keys[(l, h)]["n"] += X_k.shape[0]
                X_v = v_l[0, h, :, :hd].double()
                cov_vals[(l, h)]["xtx"] += X_v.T @ X_v
                cov_vals[(l, h)]["n"] += X_v.shape[0]
        n_done += 1
        if n_done >= n_sequences:
            break
        if n_done % max(n_sequences // 10, 1) == 0:
            print(f"    Calib: {n_done}/{n_sequences} ({time.time()-t_cal:.0f}s)")

    print(f"  Calibration: {n_done} seqs in {time.time()-t_cal:.1f}s")

    # Eigendecompose
    def eigen(cov_dict):
        res = {}
        for l in range(n_layers):
            for h in range(n_kv):
                ns = cov_dict[(l, h)]["n"]
                if ns < 2:
                    res[(l, h)] = {"d_eff": 0, "kappa": 0, "ev": [], "dims95": 0, "dims99": 0}
                    continue
                C = (cov_dict[(l, h)]["xtx"] / ns).float()
                ev, _ = torch.linalg.eigh(C)
                ev = ev.flip(0).clamp(min=0)
                s = ev.sum(); s2 = (ev**2).sum()
                d_eff = float(s**2 / s2) if s2 > 1e-12 else 1.0
                di = max(1, min(round(d_eff), hd - 1))
                kappa = float(ev[di-1] / max(ev[min(di, hd-1)], 1e-10))
                kappa = min(kappa, 1e6)
                cv = ev.cumsum(0) / max(s, 1e-12)
                d95 = int((cv < 0.95).sum().item()) + 1
                d99 = int((cv < 0.99).sum().item()) + 1
                res[(l, h)] = {"d_eff": d_eff, "kappa": kappa,
                               "ev": ev[:min(64, hd)].tolist(),
                               "dims95": min(d95, hd), "dims99": min(d99, hd)}
        return res

    ek = eigen(cov_keys); ev = eigen(cov_vals)

    import numpy as np
    all_kd = [ek[(l,h)]["d_eff"] for l in range(n_layers) for h in range(n_kv) if ek[(l,h)]["d_eff"] > 0]
    all_vd = [ev[(l,h)]["d_eff"] for l in range(n_layers) for h in range(n_kv) if ev[(l,h)]["d_eff"] > 0]
    all_kk = [ek[(l,h)]["kappa"] for l in range(n_layers) for h in range(n_kv) if ek[(l,h)]["kappa"] > 0]

    mkd = float(np.mean(all_kd)) if all_kd else 0
    mvd = float(np.mean(all_vd)) if all_vd else 0
    mkk = float(np.mean(all_kk)) if all_kk else 0

    kd_pl = [float(np.mean([ek[(l,h)]["d_eff"] for h in range(n_kv)])) for l in range(n_layers)]
    vd_pl = [float(np.mean([ev[(l,h)]["d_eff"] for h in range(n_kv)])) for l in range(n_layers)]
    kk_pl = [float(np.mean([ek[(l,h)]["kappa"] for h in range(n_kv)])) for l in range(n_layers)]
    d95_pl = [float(np.mean([ek[(l,h)]["dims95"] for h in range(n_kv)])) for l in range(n_layers)]

    rl = [0, n_layers//2, n_layers-1]
    rsp = {}
    for l in rl:
        rsp[f"layer_{l}"] = {
            "key_eigenvalues": ek[(l,0)]["ev"], "val_eigenvalues": ev[(l,0)]["ev"],
            "key_d_eff": ek[(l,0)]["d_eff"], "val_d_eff": ev[(l,0)]["d_eff"],
        }

    total_time = time.time() - t_start
    result = {
        "model": model_name, "short_name": "", "n_layers": n_layers,
        "n_kv_heads": n_kv, "head_dim": hd, "n_params_b": round(n_params, 2),
        "n_sequences": n_done, "max_length": max_length,
        "calibration_dataset": "wikitext-103-raw-v1",
        "measurement_time_s": round(total_time, 1),
        "gpu": torch.cuda.get_device_name(0),
        "key_deff": round(mkd, 4), "val_deff": round(mvd, 4),
        "kappa": round(mkk, 4), "key_pct": round(mkd / hd * 100, 2),
        "key_deff_per_layer": [round(x, 4) for x in kd_pl],
        "val_deff_per_layer": [round(x, 4) for x in vd_pl],
        "kappa_per_layer": [round(x, 4) for x in kk_pl],
        "dims95_per_layer": [round(x, 1) for x in d95_pl],
        "representative_spectra": rsp,
    }

    del model
    torch.cuda.empty_cache()
    return result


# ═══════════════════════════════════════════════════════════════
# Task 1: Measure NEW models (n=100)
# ═══════════════════════════════════════════════════════════════

@app.function(
    image=image, gpu="B200", timeout=7200,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/results": vol},
)
def measure_new_model(model_name: str, needs_token: bool, short_name: str):
    """Measure a single new model at n=100."""
    import os
    out_path = f"/results/{short_name}.json"
    if os.path.exists(out_path):
        print(f"[SKIP] {short_name} already done")
        with open(out_path) as f:
            return json.load(f)

    print(f"\n{'='*60}\n  {short_name} (n=100)\n{'='*60}")
    result = _measure_core(model_name, needs_token, n_sequences=100)
    result["short_name"] = short_name

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    vol.commit()
    print(f"  ✓ key_deff={result['key_deff']:.2f} val_deff={result['val_deff']:.2f}")
    return result


# ═══════════════════════════════════════════════════════════════
# Task 2: Calibration stability sweep (n=100, 1000, 10000)
# ═══════════════════════════════════════════════════════════════

@app.function(
    image=image, gpu="B200", timeout=14400,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/results": vol},
)
def measure_stability(model_name: str, needs_token: bool, short_name: str, n_values: list):
    """Run calibration sweep at multiple n values for one model."""
    import os
    out_path = f"/results/stability_{short_name}.json"
    if os.path.exists(out_path):
        print(f"[SKIP] stability_{short_name} already done")
        with open(out_path) as f:
            return json.load(f)

    print(f"\n{'='*60}")
    print(f"  Stability sweep: {short_name}")
    print(f"  n_values: {n_values}")
    print(f"{'='*60}")

    sweep_results = {}
    for n in n_values:
        print(f"\n  --- n={n} ---")
        result = _measure_core(model_name, needs_token, n_sequences=n)
        sweep_results[str(n)] = {
            "n_sequences": n,
            "key_deff": result["key_deff"],
            "val_deff": result["val_deff"],
            "kappa": result["kappa"],
            "key_pct": result["key_pct"],
            "key_deff_per_layer": result["key_deff_per_layer"],
            "measurement_time_s": result["measurement_time_s"],
        }
        print(f"  n={n}: key_deff={result['key_deff']:.4f}, val_deff={result['val_deff']:.4f}")

    out = {
        "model": model_name,
        "short_name": short_name,
        "head_dim": result["head_dim"],
        "n_layers": result["n_layers"],
        "n_kv_heads": result["n_kv_heads"],
        "sweep": sweep_results,
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    print(f"\n  ✓ Stability sweep saved for {short_name}")
    return out


@app.local_entrypoint()
def main():
    """Launch all measurements in parallel."""

    print(f"=== SPECTRAL INDEX: FULL SWEEP ===")
    print(f"  {len(NEW_MODELS)} new models (n=100)")
    print(f"  {len(STABILITY_MODELS)} stability sweeps (n=100,1000,10000)")
    print()

    # Launch all new model measurements in parallel
    model_handles = []
    for model_name, needs_token, short_name in NEW_MODELS:
        h = measure_new_model.spawn(model_name, needs_token, short_name)
        model_handles.append((short_name, h))
        print(f"  Spawned: {short_name}")

    # Launch stability sweeps in parallel
    stability_handles = []
    for model_name, needs_token, short_name in STABILITY_MODELS:
        h = measure_stability.spawn(model_name, needs_token, short_name, STABILITY_N_VALUES)
        stability_handles.append((short_name, h))
        print(f"  Spawned stability: {short_name}")

    print(f"\n  Waiting for {len(model_handles)} model measurements...")
    results = []
    failed = []
    for short_name, handle in model_handles:
        try:
            result = handle.get()
            results.append(result)
            print(f"  ✓ {short_name}: key_deff={result['key_deff']:.2f}")
        except Exception as e:
            print(f"  ✗ {short_name}: {e}")
            failed.append((short_name, str(e)))

    print(f"\n  Waiting for {len(stability_handles)} stability sweeps...")
    stability_results = []
    for short_name, handle in stability_handles:
        try:
            result = handle.get()
            stability_results.append(result)
            sweep = result["sweep"]
            vals = [f"n={k}: {v['key_deff']:.4f}" for k, v in sweep.items()]
            print(f"  ✓ {short_name}: {', '.join(vals)}")
        except Exception as e:
            print(f"  ✗ {short_name} stability: {e}")

    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(results)} models, {len(stability_results)} stability sweeps")
    if failed:
        print(f"Failed ({len(failed)}):")
        for name, err in failed:
            print(f"  - {name}: {err[:100]}")

    # Download all results locally
    import os
    os.makedirs("results", exist_ok=True)
    for r in results:
        path = f"results/{r['short_name']}.json"
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
    for r in stability_results:
        path = f"results/stability_{r['short_name']}.json"
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
    print("All results saved to results/")
