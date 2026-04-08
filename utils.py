"""
utils.py — Utility functions for LLM CPU inference with llama-cpp-python.
"""

import sys
import time
import threading
from pathlib import Path

import psutil
import cpuinfo
from llama_cpp import Llama


# ── System diagnostics ─────────────────────────────────────────────────────

def get_system_info() -> dict:
    """
    Returns a summary of the host system for display and decision-making.

    Returns
    -------
    dict with keys:
        cpu_brand       : str   — CPU marketing name
        physical_cores  : int   — P-core count (logical=False)
        logical_cores   : int   — total thread count (with HT)
        ram_total_gb    : float — total installed RAM
        ram_avail_gb    : float — currently available RAM
        simd_flags      : list  — detected SIMD extensions (avx2, avx512f, …)
    """
    mem  = psutil.virtual_memory()
    cpu  = cpuinfo.get_cpu_info()
    flags = cpu.get("flags", [])
    simd = [f for f in ["avx2", "avx512f", "avx512vnni", "amx_tile"] if f in flags]

    return {
        "cpu_brand":      cpu.get("brand_raw", "Unknown CPU"),
        "physical_cores": psutil.cpu_count(logical=False) or 1,
        "logical_cores":  psutil.cpu_count(logical=True)  or 1,
        "ram_total_gb":   round(mem.total     / 1024**3, 1),
        "ram_avail_gb":   round(mem.available / 1024**3, 1),
        "simd_flags":     simd,
    }


def check_llama_simd() -> tuple[dict, dict, list]:
    """
    Checks whether llama-cpp-python was compiled with the SIMD flags
    available on the host CPU.

    A mismatch (CPU supports AVX2 but build was compiled without it) is the
    single most common cause of poor inference speed (~0.2 tok/s vs 10–20 tok/s).

    Detection strategy (llama-cpp-python >= 0.3.x)
    -----------------------------------------------
    ggml_cpu_has_avx2() and friends were removed from the Python-facing API in
    0.3.x but the symbols still exist in the compiled DLL/SO. The most reliable
    detection method is to scan the binary bytes of ggml-cpu.dll (Windows) or
    libggml-cpu.so (Linux/macOS) for the flag strings that llama.cpp embeds in
    its build:
      - b'AVX2'   → compiled with AVX2
      - b'AVX512' → compiled with AVX-512

    This works regardless of pip cache, direct_url.json presence, or wheel tag.

    Returns
    -------
    cpu_has  : dict  — {flag: bool} for avx2, avx512f, amx_tile on the CPU
    build_has: dict  — {flag: bool} detected from the compiled DLL/SO bytes
    missing  : list  — SIMD names available on CPU but absent from the build
    """
    import pathlib
    import sys
    import llama_cpp as _lc

    cpu_flags = cpuinfo.get_cpu_info().get("flags", [])
    cpu_has = {f: f in cpu_flags for f in ["avx2", "avx512f", "amx_tile"]}
    build_has = {"avx2": False, "avx512f": False, "amx_tile": False}

    try:
        pkg_dir = pathlib.Path(_lc.__file__).parent

        candidates = list(pkg_dir.rglob("libggml-cpu*.so*"))
        if not candidates:
            candidates = list(pkg_dir.rglob("*.so*"))

        # Read bytes from the first (largest) candidate
        target = max(candidates, key=lambda p: p.stat().st_size)
        data = target.read_bytes()

        build_has["avx2"]   = b"AVX2"   in data or b"avx2"   in data
        build_has["avx512f"] = b"AVX512" in data or b"avx512" in data
        # AMX: no reliable string marker in current llama.cpp builds
        build_has["amx_tile"] = False

    except Exception:
        # If binary scanning fails for any reason, assume no SIMD (conservative)
        pass

    missing = [k.upper() for k, v in cpu_has.items() if v and not build_has[k]]
    return cpu_has, build_has, missing


def print_simd_fix(missing: list) -> None:
    """
    Prints installation commands to fix a SIMD mismatch on Linux.
    Call after check_llama_simd() when missing is non-empty.

    Parameters
    ----------
    missing : list of str — SIMD names absent from the build (e.g. ['AVX2'])
    """
    cmake_flags = []
    if "AVX2"    in missing: cmake_flags += ["-DGGML_AVX2=ON"]
    if "AVX512F" in missing: cmake_flags += ["-DGGML_AVX512=ON"]
    if "AMX_TILE" in missing: cmake_flags += ["-DGGML_AMX_INT8=ON", "-DGGML_AMX_BF16=ON"]
    cmake_str = " ".join(cmake_flags) if cmake_flags else "-DGGML_AVX2=ON"
    wheel_tag = "avx512" if any("AVX512" in m or "AMX" in m for m in missing) else "avx2"

    print(f"\n⚠️  SIMD flags missing in llama-cpp-python build: {missing}")
    print("   Likely cause: installed without CMAKE_ARGS → ~0.2 tok/s instead of 10–20 tok/s\n")

    print("┌─ OPTION A  Pre-built wheel (no compiler needed) ───────────────────────────")
    print(f"│  pip install llama-cpp-python \\")
    print(f"│      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/{wheel_tag}")
    print("└────────────────────────────────────────────────────────────────────────────\n")

    print("┌─ OPTION B  Compile from source ────────────────────────────────────────────")
    print(f'│  CMAKE_ARGS="{cmake_str}" \\')
    print("│    pip install llama-cpp-python --force-reinstall --no-cache-dir")
    print("└────────────────────────────────────────────────────────────────────────────\n")

    print("   → After installing, restart the kernel and re-run this cell.")


# ── GGUF file discovery ────────────────────────────────────────────────────

def find_gguf(model_name: str, gguf_dir: str | Path) -> Path:
    """
    Auto-detects the quantised GGUF file for a given model.

    Looks for files matching ``<model_folder>-*.gguf`` in gguf_dir,
    excluding the intermediate F16 file. Returns the first match.

    Parameters
    ----------
    model_name : HuggingFace model ID (e.g. "mistralai/Ministral-3B-Instruct-2412")
    gguf_dir   : directory that contains the .gguf files

    Returns
    -------
    Path — path to the detected GGUF file (may not exist if none found)
    """
    model_folder = model_name.split("/", 1)[1] if "/" in model_name else model_name
    gguf_dir = Path(gguf_dir)

    candidates = [
        f for f in gguf_dir.glob(f"{model_folder}-*.gguf")
        if not f.stem.endswith("-f16")
    ]

    if candidates:
        path = candidates[0]
        print(f"  GGUF detected: {path.name}")
        return path

    # Fallback: return the expected balanced path even if it doesn't exist yet
    fallback = gguf_dir / f"{model_folder}-Q4_K_M.gguf"
    print(f"  No GGUF found — expected: {fallback.name}")
    return fallback


# ── Thread selection ────────────────────────────────────────────────────────

def get_inference_threads() -> tuple[int, int]:
    """
    Returns the optimal (n_threads, n_threads_batch) for the host CPU.

    Rules
    -----
    - Use all physical cores minus 1 to leave headroom for the OS.
      Works on any CPU: Intel (hybrid or not), AMD Ryzen/EPYC, ARM.
    - Never use logical/hyperthreaded cores: HT adds memory pressure without
      speed gain for matrix-vector workloads like LLM inference.
    - n_threads_batch (prompt processing) can use all physical cores since it
      is a one-shot parallel operation, not latency-sensitive.

    Source: https://www.mintlify.com/ggml-org/llama.cpp/advanced/performance-tuning

    Returns
    -------
    n_threads       : int — threads for token generation  (physical - 1)
    n_threads_batch : int — threads for prompt processing (all physical)
    """
    physical = psutil.cpu_count(logical=False) or 1
    n_threads = max(1, physical - 1)

    print(f"  CPU: {physical} physical cores  →  inference: {n_threads} threads  |  batch: {physical} threads")
    return n_threads, physical


# ── Model loading ───────────────────────────────────────────────────────────

def load_model(
    gguf_path: str | Path,
    n_threads: int | None = None,
    n_threads_batch: int | None = None,
    n_ctx: int | None = None,
    n_batch: int = 512,
    verbose: bool = False,
) -> tuple[Llama, dict]:
    """
    Loads a GGUF model with optimised settings for CPU inference.

    Automatically applies:
    - flash_attn=True      — reduces memory for long contexts (llama.cpp >= b3770)
    - KV cache Q8_0        — halves KV RAM when n_ctx > 4096, near-zero quality loss
      (type_k=8, type_v=8; available in llama-cpp-python >= 0.2.83)
    - use_mmap=True        — memory-map the file; pages loaded on demand by the OS

    Falls back gracefully if the installed version does not support these parameters.

    Parameters
    ----------
    gguf_path        : path to the .gguf file
    n_threads        : inference threads; auto-detected if None
    n_threads_batch  : batch (prompt) threads; auto-detected if None
    n_ctx            : context length in tokens; auto-detected from RAM if None
    n_batch          : batch size (default 512)
    verbose          : show llama.cpp loading logs (default False)

    Returns
    -------
    model     : Llama  — loaded model instance
    load_info : dict   — {load_time_s, ram_used_gb, n_threads, n_threads_batch, n_ctx}
    """
    gguf_path = Path(gguf_path)

    if n_threads is None or n_threads_batch is None:
        _t, _b = get_inference_threads()
        n_threads       = n_threads       if n_threads       is not None else _t
        n_threads_batch = n_threads_batch if n_threads_batch is not None else _b

    if n_ctx is None:
        ram_avail = psutil.virtual_memory().available / 1024**3
        n_ctx = 4096 if ram_avail > 20 else 2048
        print(f"  Context (RAM heuristic): {n_ctx} tokens")
    else:
        print(f"  Context: {n_ctx} tokens")

    print(f"  Threads — inference: {n_threads}  |  batch: {n_threads_batch}")

    base_kwargs = dict(
        model_path=str(gguf_path),
        n_threads=n_threads,
        n_threads_batch=n_threads_batch,
        n_ctx=n_ctx,
        n_batch=n_batch,
        use_mmap=True,
        use_mlock=False,
        verbose=verbose,
    )

    ram_before = psutil.virtual_memory().used / 1024**3
    t0 = time.time()

    # Fallback chain: try flash_attn + KV Q4_0 first (fastest / least RAM),
    # then Q8_0 (near-lossless), then no flash_attn, then minimal params.
    # type_k=2 = Q4_0 (fast), type_k=8 = Q8_0 (accurate)
    model = None
    for kv_type, label in [(2, "Q4_0"), (8, "Q8_0"), (None, None)]:
        try:
            if kv_type is not None:
                model = Llama(**base_kwargs, flash_attn=True, type_k=kv_type, type_v=kv_type)
                print(f"  Flash Attention + KV cache {label} enabled")
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

    load_info = {
        "load_time_s":      round(load_time, 1),
        "ram_used_gb":      round(ram_used,  1),
        "n_threads":        n_threads,
        "n_threads_batch":  n_threads_batch,
        "n_ctx":            n_ctx,
    }

    print(f"\nModel loaded in {load_time:.1f}s  |  +{ram_used:.1f} GB RAM  |  ctx={n_ctx} tokens")
    return model, load_info


# ── Prompt formatting ───────────────────────────────────────────────────────

def format_prompt(model_name: str, system: str, user: str) -> str:
    """
    Auto-detects the prompt template from the model family name.

    Each model family expects a different chat format — using the wrong one
    silently degrades output quality (the model does not "see" the system prompt).

    Supported families
    ------------------
    - ChatML  (Qwen, Mistral, Ministral, Phi-4, Nanbeige) : <|im_start|>
    - Llama-3 (Meta Llama, RNJ, EssentialAI)              : <|begin_of_text|>
    - Gemma   (Google Gemma 2/4)                           : <start_of_turn>
    - Generic fallback                                      : ### User / ### Assistant

    Parameters
    ----------
    model_name : HuggingFace model ID or local folder name (used for family detection)
    system     : system prompt text
    user       : user message text

    Returns
    -------
    str — fully formatted prompt ready to pass to model()
    """
    name = model_name.lower()

    if any(x in name for x in ["qwen", "mistral", "ministral", "phi-4", "nanbeige"]):
        # ChatML format — used by Qwen, Mistral, Ministral, Phi-4, Nanbeige
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    elif any(x in name for x in ["llama", "rnj"]):
        # Llama-3 instruct format — used by Meta Llama 3, RNJ-1
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}"
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user}"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    elif "gemma" in name:
        # Gemma has no native system prompt — prepend it to the first user turn
        return (
            f"<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
    else:
        # Generic fallback for unknown model families
        return f"### System:\n{system}\n\n### User:\n{user}\n\n### Assistant:\n"


def get_stop_tokens(model_name: str) -> list[str]:
    """
    Returns the EOS / end-of-turn stop tokens for a given model family.

    Pass the returned list as the `stop` parameter in model() to prevent
    the model from generating past the end of its response.

    Parameters
    ----------
    model_name : HuggingFace model ID or local folder name

    Returns
    -------
    list of str — stop tokens; empty list if family is unknown (no early stop)
    """
    name = model_name.lower()
    if any(x in name for x in ["qwen", "mistral", "ministral", "phi-4", "nanbeige"]):
        return ["<|im_end|>", "<|im_start|>"]
    elif any(x in name for x in ["llama", "rnj"]):
        return ["<|eot_id|>", "<|end_of_text|>"]
    elif "gemma" in name:
        return ["<end_of_turn>"]
    return []


# ── Inference ───────────────────────────────────────────────────────────────

def generate(
    model: Llama,
    model_name: str,
    prompt: str,
    system_prompt: str = (
        "You are a helpful, concise assistant. "
        "Answer clearly and directly. "
        "If the question is about code, provide working examples."
    ),
    max_tokens: int = 150,
    temperature: float = 0.3,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
    stream: bool = True,
) -> dict:
    """
    Runs inference and returns text + performance metrics.

    A background thread monitors CPU utilisation during generation.
    When stream=True, tokens are printed to stdout in real time.

    Parameters
    ----------
    model          : loaded Llama instance (from load_model)
    model_name     : HuggingFace model ID — used to select prompt format and stop tokens
    prompt         : user message text
    system_prompt  : system role text (default: helpful assistant)
    max_tokens     : maximum tokens to generate
    temperature    : sampling temperature (0 = deterministic, 1 = creative)
    top_p          : nucleus sampling probability
    repeat_penalty : penalty for repeating tokens (> 1 discourages repetition)
    stream         : if True, print tokens as they are generated

    Returns
    -------
    dict with keys:
        text        : str   — full generated text
        tokens      : int   — number of completion tokens
        time_s      : float — wall-clock time in seconds
        tok_s       : float — tokens per second
        avg_cpu_pct : float — average CPU utilisation during generation (%)
    """
    cpu_samples = []
    stop_flag   = threading.Event()

    def _monitor():
        while not stop_flag.is_set():
            cpu_samples.append(psutil.cpu_percent(interval=0.2))

    threading.Thread(target=_monitor, daemon=True).start()

    formatted   = format_prompt(model_name, system_prompt, prompt)
    stop_tokens = get_stop_tokens(model_name)
    t0 = time.time()

    if stream:
        chunks = model(
            formatted,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop_tokens,
            echo=False,
            stream=True,
        )
        text = ""
        completion_tokens = 0
        for chunk in chunks:
            delta = chunk["choices"][0]["text"]
            text += delta
            completion_tokens += 1
            sys.stdout.write(delta)
            sys.stdout.flush()
        print()  # newline after streamed output
    else:
        resp = model(
            formatted,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop_tokens,
            echo=False,
        )
        text = resp["choices"][0]["text"]
        completion_tokens = resp["usage"]["completion_tokens"]

    elapsed = time.time() - t0
    stop_flag.set()

    # Estimate prefill vs generation split.
    # Prefill (prompt processing) is ~3× faster per token than generation.
    # Approximation: prefill_time ≈ total × (prompt_tokens / total_tokens) × 0.3
    if not stream:
        prompt_tokens = resp["usage"]["prompt_tokens"]
        total_tokens  = resp["usage"]["total_tokens"]
        prefill_frac  = (prompt_tokens / total_tokens * 0.3) if total_tokens > 0 else 0
    else:
        prefill_frac = 0.0  # not available without usage stats in stream mode

    prefill_time    = round(elapsed * prefill_frac, 2)
    generation_time = round(elapsed - prefill_time, 2)
    tok_s           = completion_tokens / generation_time if generation_time > 0 else 0
    avg_cpu         = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0

    return {
        "text":            text,
        "tokens":          completion_tokens,
        "time_s":          round(elapsed, 2),
        "prefill_time_s":  prefill_time,
        "generation_time_s": generation_time,
        "tok_s":           round(tok_s, 1),
        "avg_cpu_pct":     round(avg_cpu, 1),
    }


def warmup(
    model: Llama,
    model_name: str,
    system_prompt: str = "You are a helpful assistant.",
) -> None:
    """
    Runs a silent single-token warmup call.

    The first inference call is always slower because the OS must:
    1. Page in the model weights from disk (if using mmap)
    2. Allocate the KV cache buffers
    3. JIT-compile any lazy kernels

    Call warmup() before any benchmark to ensure measurements are representative.

    Parameters
    ----------
    model         : loaded Llama instance
    model_name    : HuggingFace model ID (for prompt format detection)
    system_prompt : short system text (content does not matter for warmup)
    """
    print("Warming up...", end=" ", flush=True)
    model(
        format_prompt(model_name, system_prompt, "Hi"),
        max_tokens=1,
    )
    print("ready.")