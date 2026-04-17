"""
utils.py — Utility functions for LLM CPU inference with llama-cpp-python.
"""

import os
import sys
import time
import threading
from pathlib import Path

import psutil
import cpuinfo
from llama_cpp import Llama


# ── System diagnostics ─────────────────────────────────────────────────────

def get_system_info() -> dict:
    """Returns CPU, RAM, and SIMD info for the host system."""
    mem  = psutil.virtual_memory()
    cpu  = cpuinfo.get_cpu_info()
    flags = cpu.get("flags", [])
    simd = [f for f in ["avx2", "avx512f", "avx512vnni", "amx_tile"] if f in flags]

    return {
        "cpu_brand":      cpu.get("brand_raw", "Unknown CPU"),
        "physical_cores": psutil.cpu_count(logical=False) or os.cpu_count() or 1,
        "logical_cores":  psutil.cpu_count(logical=True)  or os.cpu_count() or 1,
        "ram_total_gb":   round(mem.total     / 1024**3, 1),
        "ram_avail_gb":   round(mem.available / 1024**3, 1),
        "simd_flags":     simd,
    }


def check_llama_simd() -> tuple[dict, dict, list]:
    """
    Checks whether llama-cpp-python was compiled with the SIMD flags available on the CPU.

    A mismatch (CPU has AVX2, build uses SSE2) is the #1 cause of slow inference:
    ~0.2 tok/s instead of 10–20 tok/s.

    Returns
    -------
    cpu_has  : dict  — {flag: bool} for avx2, avx512f, amx_tile
    build_has: dict  — {flag: bool} detected from the compiled .so bytes
    missing  : list  — SIMD flags available on CPU but absent from the build
    """
    import llama_cpp as _lc

    cpu_flags = cpuinfo.get_cpu_info().get("flags", [])
    cpu_has   = {f: f in cpu_flags for f in ["avx2", "avx512f", "amx_tile"]}
    build_has = {"avx2": False, "avx512f": False, "amx_tile": False}

    try:
        pkg_dir    = Path(_lc.__file__).parent
        candidates = list(pkg_dir.rglob("libggml-cpu*.so*")) or list(pkg_dir.rglob("*.so*"))
        target     = max(candidates, key=lambda p: p.stat().st_size)
        data       = target.read_bytes()

        build_has["avx2"]    = b"AVX2"   in data or b"avx2"   in data
        build_has["avx512f"] = b"AVX512" in data or b"avx512" in data
    except Exception:
        pass

    missing = [k.upper() for k, v in cpu_has.items() if v and not build_has[k]]
    return cpu_has, build_has, missing


def print_simd_fix(missing: list) -> None:
    """Prints install commands to fix a SIMD mismatch."""
    cmake_flags = []
    if "AVX2"     in missing: cmake_flags += ["-DGGML_AVX2=ON"]
    if "AVX512F"  in missing: cmake_flags += ["-DGGML_AVX512=ON"]
    if "AMX_TILE" in missing: cmake_flags += ["-DGGML_AMX_INT8=ON", "-DGGML_AMX_BF16=ON"]
    cmake_str = " ".join(cmake_flags) if cmake_flags else "-DGGML_AVX2=ON"
    wheel_tag = "avx512" if any("AVX512" in m or "AMX" in m for m in missing) else "avx2"

    print(f"\n⚠️  SIMD flags missing in build: {missing}")
    print("   Cause: installed without CMAKE_ARGS → ~0.2 tok/s instead of 10–20 tok/s\n")
    print(f"Option A (pre-built wheel):  pip install llama-cpp-python "
          f"--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/{wheel_tag}")
    print(f'Option B (compile):          CMAKE_ARGS="{cmake_str}" '
          f"pip install llama-cpp-python --force-reinstall --no-cache-dir")
    print("\n→ Restart the kernel after reinstalling.")


# ── GGUF file discovery ────────────────────────────────────────────────────

def find_gguf(model_name: str, gguf_dir: str | Path) -> Path:
    """
    Auto-detects the quantised GGUF file for a given model.

    Searches for ``<model_folder>-*.gguf`` in gguf_dir, excluding the F16 file.
    Falls back to the expected Q4_K_M path if nothing is found.
    """
    model_folder = model_name.split("/", 1)[1] if "/" in model_name else model_name
    gguf_dir     = Path(gguf_dir)
    candidates   = [f for f in gguf_dir.glob(f"{model_folder}-*.gguf")
                    if not f.stem.endswith("-f16")]

    if candidates:
        path = candidates[0]
        print(f"  GGUF detected: {path.name}")
        return path

    fallback = gguf_dir / f"{model_folder}-Q4_K_M.gguf"
    print(f"  No GGUF found — expected: {fallback.name}")
    return fallback


# ── Model loading ───────────────────────────────────────────────────────────

def load_model(
    gguf_path: str | Path,
    n_threads: int | None = None,
    n_ctx: int | None = None,
    n_batch: int = 512,
    verbose: bool = False,
) -> tuple[Llama, dict]:
    """
    Loads a GGUF model with optimised CPU settings.

    Thread count: all physical cores (os.cpu_count() fallback for WSL).
    Flash Attention + KV Q4_0: tried first; falls back gracefully if unsupported.
    Context: auto-detected from available RAM if not specified.

    Returns
    -------
    model     : Llama
    load_info : dict — {load_time_s, ram_used_gb, n_threads, n_ctx}
    """
    gguf_path = Path(gguf_path)

    # Use all physical cores — leave 0 for OS (llama.cpp manages internal scheduling).
    # os.cpu_count() is the WSL-safe fallback: psutil may under-report under WSL.
    if n_threads is None:
        n_threads = psutil.cpu_count(logical=False) or os.cpu_count() or 1

    if n_ctx is None:
        ram_avail = psutil.virtual_memory().available / 1024**3
        n_ctx = 4096 if ram_avail > 20 else 2048

    print(f"  Threads: {n_threads}  |  ctx: {n_ctx} tokens")

    base_kwargs = dict(
        model_path=str(gguf_path),
        n_threads=n_threads,
        n_threads_batch=n_threads,
        n_ctx=n_ctx,
        n_batch=n_batch,
        use_mmap=True,
        use_mlock=True,
        verbose=verbose,
    )

    ram_before = psutil.virtual_memory().used / 1024**3
    t0 = time.time()

    # Fallback chain: Q4_0 KV (fastest) → Q8_0 KV (accurate) → no flash_attn
    model = None
    fallbacks = [
                (2, "Q4_0"),   # meilleur si supporté
                (4, "Q4_K"),   # plus compatible
                (8, "Q8_0"),   # lent mais stable
                (None, None),  # sans flash_attn
            ]
    for kv_type, label in fallbacks:
        try:
            if kv_type is not None:
                model = Llama(**base_kwargs, flash_attn=True, type_k=kv_type, type_v=kv_type)
                print(f"  Flash Attention + KV {label} enabled")
            else:
                model = Llama(**base_kwargs)
                print("  ⚠️  flash_attn not supported — upgrade llama-cpp-python >= 0.3.4")
            break
        except (TypeError, ValueError):
            continue

    if model is None:
        model = Llama(model_path=str(gguf_path), n_threads=n_threads, n_ctx=n_ctx, verbose=verbose)

    load_time = time.time() - t0
    ram_used  = psutil.virtual_memory().used / 1024**3 - ram_before

    print(f"\n  Loaded in {load_time:.1f}s  |  +{ram_used:.1f} GB RAM")
    return model, {"load_time_s": round(load_time, 1), "ram_used_gb": round(ram_used, 1),
                   "n_threads": n_threads, "n_ctx": n_ctx}


# ── Inference ───────────────────────────────────────────────────────────────

def generate(
    model: Llama,
    prompt: str,
    system_prompt: str = "You are a helpful, concise assistant. Answer clearly and directly.",
    max_tokens: int = 150,
    temperature: float = 0.3,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
    stream: bool = True,
) -> dict:
    """
    Runs inference using the model's built-in chat template (create_chat_completion).

    No manual prompt formatting needed — the GGUF contains the chat template.
    A background thread monitors CPU usage during generation.

    Returns
    -------
    dict: text, tokens, time_s, prefill_time_s, generation_time_s, tok_s, avg_cpu_pct
    """
    cpu_samples = []
    stop_flag   = threading.Event()

    def _monitor():
        while not stop_flag.is_set():
            cpu_samples.append(psutil.cpu_percent(interval=0.2))

    threading.Thread(target=_monitor, daemon=True).start()
    t0 = time.time()

    messages = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": prompt},
    ]
    common_kwargs = dict(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,   
    )

    if stream:
        text, completion_tokens = "", 0
        for chunk in model.create_chat_completion(stream=True, **common_kwargs):
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                text += delta
                completion_tokens += 1
                sys.stdout.write(delta)
                sys.stdout.flush()
        print()
        prompt_tokens = 0  # not available in stream mode
    else:
        resp              = model.create_chat_completion(**common_kwargs)
        text              = resp["choices"][0]["message"]["content"]
        completion_tokens = resp["usage"]["completion_tokens"]
        prompt_tokens     = resp["usage"]["prompt_tokens"]

    elapsed = time.time() - t0
    stop_flag.set()

    # Prefill (prompt processing) is ~3× faster per token than generation.
    total_tokens    = completion_tokens + prompt_tokens
    prefill_frac    = (prompt_tokens / total_tokens * 0.3) if total_tokens > 0 else 0
    prefill_time    = round(elapsed * prefill_frac, 2)
    generation_time = round(elapsed - prefill_time, 2)
    tok_s           = completion_tokens / generation_time if generation_time > 0 else 0
    avg_cpu         = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0

    return {
        "text":              text,
        "tokens":            completion_tokens,
        "time_s":            round(elapsed, 2),
        "prefill_time_s":    prefill_time,
        "generation_time_s": generation_time,
        "tok_s":             round(tok_s, 1),
        "avg_cpu_pct":       round(avg_cpu, 1),
    }


def warmup(model: Llama) -> None:
    """
    Silent single-token call to pre-allocate KV cache and page in weights.
    Call before benchmarking so the first measured inference is representative.
    """
    print("Warming up...", end=" ", flush=True)
    model.create_chat_completion(
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1,
    )
    print("ready.")
