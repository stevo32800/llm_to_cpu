"""
Script to optimise a HuggingFace model for CPU inference using llama.cpp.
"""

import subprocess
from pathlib import Path
import cpuinfo
import psutil
import json
import shutil
import sys 

class cpu_optimizer:
    """
    Automates the conversion of a Hugging Face model to an optimised GGUF format for CPU inference.

    Steps:
    1. Clone the llama.cpp repository if it does not already exist
    2. Build llama.cpp with optimisation flags suited to the host CPU
    3. Convert the Hugging Face model to GGUF format at F16 precision
    4. Quantise the model for the best quality / speed / RAM trade-off
    5. Warn if the chosen quantisation level may noticeably degrade quality
    6. Delete the intermediate F16 file to reclaim disk space
    """

    # Actual bits per parameter — values measured on an 8B Llama model.
    # K-quant formats (Q4_K_M, Q5_K_M, Q6_K) use hierarchical super-blocks
    # where critical layers are promoted to higher precision, hence values
    # slightly above the nominal bit count.
    # Source: https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
    BITS_PER_PARAM = {
        "F16":     16.00,
        "Q8_0":     8.50,
        "Q6_K":     6.56,
        "Q5_K_M":   5.70,
        "Q4_K_M":   4.89,
        "Q3_K_M":   3.74,
        "Q2_K":     3.35,
    }

    def __init__(self, model_name="Qwen/Qwen3-14B", input_path="/models/full/", output_path="/models/gguf/",
                 quant_type="Q4_K_M", quantization_verbose=True):

        self.model_name = model_name
        self.input_path = input_path
        self.output_path = output_path
        self.quant_type = (quant_type or "Q4_K_M").upper()
        self.quantization_verbose = quantization_verbose

        # Paths
        self.llama_dir = Path("/models/llama.cpp")
        model_folder = model_name.split("/", 1)[1] if "/" in model_name else model_name
        self.hf_model = Path(input_path) / model_name
        self.gguf_dir = Path(output_path)
        self.gguf_f16 = self.gguf_dir / f"{model_folder}-f16.gguf"
        self.gguf_quantized = self.gguf_dir / f"{model_folder}-{self.quant_type}.gguf"

        # Auto-detected model parameters
        self.model_params_b = None
        self.number_of_layers = None
        self.number_of_heads = None
        self.head_dim = None
        self.model_max_ctx = None
        self._params_detected = False  # Detect only once

    def _update_quantized_path(self):
        """Update the quantised output path when quant_type changes."""
        model_folder = self.model_name.split("/", 1)[1] if "/" in self.model_name else self.model_name
        self.gguf_quantized = self.gguf_dir / f"{model_folder}-{self.quant_type}.gguf"

    def _get_cpu_info(self):
        """Return CPU info (cached — cpuinfo is slow, ~2 s)."""
        if not hasattr(self, '_cpu_info_cache'):
            self._cpu_info_cache = cpuinfo.get_cpu_info()
        return self._cpu_info_cache

    def _run(self, cmd, quiet=False, **kwargs):
        """Run a shell command and raise an exception on failure."""
        print(f"\n>>> {cmd}")
        if quiet:
            kwargs.update({"stdout": subprocess.DEVNULL, "stderr": subprocess.STDOUT})
            subprocess.run(cmd, shell=True, check=True, **kwargs)
        else:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='replace', **kwargs)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd)

    def _estimate_model_ram_gb(self, quant_type=None):
        """Estimate model RAM usage for the given quantisation type."""
        quant = (quant_type or self.quant_type).upper()
        bits = self.BITS_PER_PARAM[quant]
        return (self.model_params_b * 1e9 * bits) / 8 / 1024**3 * 1.1

    def _warn_quality_tradeoff(self, ram_avail_gb):
        """Warn if the chosen quantisation may degrade output quality."""
        if self.quant_type in {"Q3_K_M", "Q2_K"}:
            print("⚠️  Aggressive quantisation: quality reduced (more errors / hallucinations possible).")
            print("   Recommendation: use 'balanced' or 'precision' profile if RAM allows.")
        elif self.quant_type == "Q4_K_M" and self.model_params_b >= 12 and ram_avail_gb < 24:
            print("ℹ️  Acceptable trade-off, but a smaller model will often give better perceived quality.")

    def _select_quant_type(self):
        """Validate and apply the requested quantisation type."""
        if self.quant_type not in self.BITS_PER_PARAM:
            raise ValueError(
                f"Unknown quantisation: {self.quant_type}. "
                f"Supported values: {', '.join(self.BITS_PER_PARAM.keys())}"
            )
        self._update_quantized_path()
        ram_avail = psutil.virtual_memory().available / 1024**3
        print(f"\n🎯 Quantisation: {self.quant_type}")
        self._warn_quality_tradeoff(ram_avail)
        return self.quant_type

    def _auto_detect_model_params(self):
        """Auto-detect model parameters from config.json."""
        if self._params_detected:
            return

        config_path = self.hf_model / "config.json"

        if not config_path.exists():
            print("⚠️  config.json not found — using default values.")
            self._set_default_params()
            self._params_detected = True
            return

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            print("\n🔍 Auto-detecting model parameters from config.json...")

            self.number_of_layers = self.number_of_layers or config.get('num_hidden_layers', 40)
            self.number_of_heads  = self.number_of_heads  or config.get('num_attention_heads', 8)
            # GQA: use num_key_value_heads for KV cache (smaller than num_attention_heads)
            self.num_kv_heads = getattr(self, 'num_kv_heads', None) or config.get('num_key_value_heads', self.number_of_heads)

            hidden_size = config.get('hidden_size', 5120)
            self.head_dim = self.head_dim or (hidden_size // self.number_of_heads)

            self.model_max_ctx = self.model_max_ctx or config.get(
                'max_position_embeddings',
                config.get('max_sequence_length', config.get('n_positions', 131072))
            )

            if self.model_params_b is None:
                # 1. Try reading from config
                num_params = config.get('num_parameters')
                if num_params:
                    self.model_params_b = num_params / 1e9
                else:
                    # 2. Try extracting from model name (e.g. "Qwen3-14B")
                    import re
                    model_size_match = re.search(r'(\d+\.?\d*)B', self.model_name, re.IGNORECASE)
                    if model_size_match:
                        self.model_params_b = float(model_size_match.group(1))
                        print(f"   Size extracted from model name: {self.model_params_b}B")
                    else:
                        # 3. Estimate with improved formula using GQA
                        vocab_size = config.get('vocab_size', 152064)
                        intermediate_size = config.get('intermediate_size', hidden_size * 4)
                        num_kv_heads = config.get('num_key_value_heads', self.number_of_heads)

                        # Attention: Q, K, V + output projection
                        attn_params = hidden_size * (
                            hidden_size
                            + 2 * (hidden_size // self.number_of_heads) * num_kv_heads
                            + hidden_size
                        )
                        # MLP: gate, up, down projections
                        mlp_params = 3 * hidden_size * intermediate_size
                        # Layer norms
                        norm_params = 2 * hidden_size

                        params_per_layer = attn_params + mlp_params + norm_params
                        embedding_params = vocab_size * hidden_size * 2  # input + output embeddings
                        total_params = self.number_of_layers * params_per_layer + embedding_params
                        self.model_params_b = round(total_params / 1e9)

            print("✅ Detected parameters:")
            print(f"   Parameters : ~{self.model_params_b}B")
            print(f"   Layers     : {self.number_of_layers}")
            print(f"   Heads      : {self.number_of_heads}")
            print(f"   Head dim   : {self.head_dim}")
            print(f"   Max context: {self.model_max_ctx:,} tokens ({self.model_max_ctx/1000:.0f}k)")

            self._params_detected = True

        except Exception as e:
            print(f"⚠️  Error reading config: {e}")
            print("   Using default values.")
            self._set_default_params()
            self._params_detected = True

    def _set_default_params(self):
        """Set default model parameters when auto-detection fails."""
        self.model_params_b   = self.model_params_b   or 14
        self.number_of_layers = self.number_of_layers or 40
        self.number_of_heads  = self.number_of_heads  or 8
        self.num_kv_heads     = getattr(self, 'num_kv_heads', None) or 8
        self.head_dim         = self.head_dim         or 128
        self.model_max_ctx    = self.model_max_ctx    or 131072

    def _check_model_exists(self):
        """Verify that the source model exists on disk."""
        if not self.hf_model.exists():
            print(f"❌ Error: model {self.hf_model} does not exist.")
            print("   Download it first with:")
            print(f"   python main_optimizer.py {self.model_name}")
            print(f"\n   Available models in {self.input_path}:")
            full_dir = Path(self.input_path)
            if full_dir.exists():
                for f in full_dir.iterdir():
                    if f.is_dir():
                        print(f"   - {f.name}")
            return False
        return True

    def _check_quantized_exists(self):
        """Check whether the quantised model already exists."""
        if self.gguf_quantized.exists():
            file_size = self.gguf_quantized.stat().st_size / 1024**3
            print("\n✅ Quantised model already present!")
            print(f"📂 File : {self.gguf_quantized.name}")
            print(f"📊 Size : {file_size:.2f} GB")
            print("\n⏭️  Quantisation skipped.")
            return True
        return False

    def _setup_directories(self):
        """Create required output directories."""
        self.gguf_dir.mkdir(parents=True, exist_ok=True)
        print(f"📂 Source model : {self.hf_model.name}")
        print(f"📂 GGUF output  : {self.gguf_dir.name}")
        print(f"📂 llama.cpp    : {self.llama_dir.name}")

    def _analyze_ram(self):
        """Analyse available RAM and compute the optimal context length."""
        ram = psutil.virtual_memory()
        ram_total = ram.total / 1024**3
        ram_avail = ram.available / 1024**3

        print(f"\n💾 Total RAM    : {ram_total:.1f} GB")
        print(f"💾 Available RAM: {ram_avail:.1f} GB")

        model_ram_gb = self._estimate_model_ram_gb()
        ram_after_model = ram_avail - model_ram_gb

        print(f"\n📊 Model {self.model_params_b}B {self.quant_type}")
        print(f"  Estimated size    : {model_ram_gb:.1f} GB")
        print(f"  RAM after model   : ~{ram_after_model:.1f} GB")

        if ram_after_model < 2:
            print(f"⚠️  Insufficient RAM! Model: {model_ram_gb:.1f} GB, available: {ram_avail:.1f} GB")

        # KV cache: 2 bytes per element (FP16), K and V
        ram_for_ctx = max(ram_after_model - 1.5, 0.5)
        num_kv_heads = getattr(self, 'num_kv_heads', self.number_of_heads)
        kv_per_token = self.number_of_layers * num_kv_heads * self.head_dim * 2 * 2
        kv_gb_per_token = kv_per_token / 1024**3
        max_ctx_by_ram = int(ram_for_ctx / kv_gb_per_token)

        ctx = min(max_ctx_by_ram, self.model_max_ctx)
        ctx = max(ctx, 512)
        ctx = (ctx // 512) * 512

        kv_cache_gb = ctx * kv_gb_per_token
        print(f"  KV cache estimate : {kv_cache_gb:.2f} GB")
        print(f"  Context           : {ctx} tokens ({ctx/1000:.0f}k)")

        return ctx

    def _detect_p_cores(self) -> int:
        """
        Return the number of P-cores (Performance cores) on hybrid Intel CPUs (12th gen+).
        On homogeneous CPUs, return the total physical core count.

        E-cores are slower for sequential inference — excluding them improves
        tok/s latency on 12th–14th gen i5/i7/i9 processors.
        """
        cores_physical = psutil.cpu_count(logical=False) or 1
        info = self._get_cpu_info()
        brand = info.get('brand_raw', '').lower()

        # Heuristic: on 12th–14th gen Intel, P-cores ≈ 20% of physical cores
        # (e.g. i7-1265U: 2P + 8E = 10 physical → return 2)
        import re
        gen_match = re.search(r'(1[234])th gen', brand)
        is_intel_hybrid = bool(gen_match) and 'intel' in brand

        if is_intel_hybrid:
            p_cores = max(2, round(cores_physical * 0.2))
            print(f"  Intel hybrid architecture detected ({gen_match.group(0)})")
            print(f"  Estimated P-cores: {p_cores} / {cores_physical} physical cores")
            print("  → Inference on P-cores only (best tok/s latency)")
            return p_cores

        return cores_physical

    def _detect_cpu_capabilities(self):
        """Detect SIMD capabilities of the host CPU."""
        info = self._get_cpu_info()
        flags = info.get('flags', [])

        has_avx2       = 'avx2'       in flags
        has_avx512     = 'avx512f'    in flags
        has_avx512vnni = 'avx512vnni' in flags
        has_amx        = 'amx_tile'   in flags

        print(f"\n🖥️  CPU: {info['brand_raw']}")
        print(f"  AVX2        : {'✓' if has_avx2       else '✗'}")
        print(f"  AVX-512     : {'✓' if has_avx512     else '✗'}")
        print(f"  AVX-512VNNI : {'✓' if has_avx512vnni else '✗'}")
        print(f"  AMX         : {'✓' if has_amx        else '✗'}")

        cores_physical = psutil.cpu_count(logical=False) or 1
        cores_logical  = psutil.cpu_count(logical=True)  or cores_physical

        print(f"\n  Physical cores : {cores_physical}")
        print(f"  Logical cores  : {cores_logical}")

        # Base flags present on all builds: native optimisation + OpenMP
        base_flags =  "-DGGML_NATIVE=ON -DGGML_OPENMP=ON -DGGML_F16C=ON"

        # Explicit SIMD flags based on detected capabilities
        if has_amx:
            simd_flags = "-DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON -DGGML_AMX_INT8=ON -DGGML_AMX_BF16=ON"
        elif has_avx512vnni:
            simd_flags = "-DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON"
        elif has_avx512:
            simd_flags = "-DGGML_AVX512=ON"
        elif has_avx2:
            simd_flags = "-DGGML_AVX=ON -DGGML_AVX2=ON"
        else:
            simd_flags = ""

        build_flags = f"{base_flags} {simd_flags}".strip()

        # Build threads: conservative fraction to avoid saturating the system
        build_threads = (
            max(8, int(cores_physical * 0.6)) if cores_physical >= 16 else
            max(4, cores_physical - 2)         if cores_physical >= 8  else
            max(1, cores_physical - 1)
        )

        # Quantisation threads: all logical cores (CPU-bound task)
        quant_threads = cores_logical

        # LTO (Link Time Optimization) significantly slows builds on small CPUs.
        # Enable only on machines with 8+ P-cores (i9, Xeon, etc.).
        lto = cores_physical >= 8

        print(f"\n  Build flags   : {build_flags}")
        print(f"  LTO           : {'ON' if lto else 'OFF (mobile CPU detected)'}")
        print(f"  Build threads : {build_threads}")
        print(f"  Quant threads : {quant_threads}")

        return build_flags, build_threads, quant_threads, cores_physical, lto

    def _clone_llama_cpp(self):
        """Clone or update the llama.cpp repository. Returns True if changes were pulled."""
        if not self.llama_dir.exists():
            print("\n📥 Cloning llama.cpp...")
            self.llama_dir.parent.mkdir(parents=True, exist_ok=True)
            self._run("git clone --depth=1 https://github.com/ggml-org/llama.cpp", cwd=self.llama_dir.parent)
            return True

        print("\n🔄 Updating llama.cpp...")
        result_before = subprocess.run(
            "git rev-parse HEAD", shell=True, check=True,
            capture_output=True, text=True, cwd=self.llama_dir
        ).stdout.strip()
        try:
            self._run("git fetch --depth=1 origin", cwd=self.llama_dir)
            self._run("git reset --hard FETCH_HEAD", cwd=self.llama_dir)
            result_after = subprocess.run(
                "git rev-parse HEAD", shell=True, check=True,
                capture_output=True, text=True, cwd=self.llama_dir
            ).stdout.strip()
            updated = result_before != result_after
            if updated:
                print(f"   New commit: {result_after[:8]} (previous: {result_before[:8]})")
            else:
                print(f"   Already up to date ({result_after[:8]}).")
            return updated
        except subprocess.CalledProcessError:
            print(f"   ⚠️  Cannot reach GitHub (proxy / network issue). Using existing local repo ({result_before[:8]}).")
            return False

    def _get_quantize_bin(self) -> Path:
        """Return the path to the llama-quantize binary."""
        return self.llama_dir / "build/bin/llama-quantize"

    def _build_llama_cpp(self, build_flags, build_threads, force=False, lto=True):
        """Build llama.cpp with the detected optimisation flags."""
        quantize_bin = self._get_quantize_bin()
        if quantize_bin.exists() and not force:
            print("\n✅ llama.cpp already built — skipping compilation.")
            return
        if force and quantize_bin.exists():
            print("\n🔄 Recompiling after llama.cpp update...")
        else:
            print("\n🔨 Building llama.cpp...")

        lto_flag = "ON" if lto else "OFF"
        cmake_configure = (
            f"cmake -B build {build_flags} "
            f"-DLLAMA_LTO={lto_flag} -DGGML_CCACHE=OFF -DCMAKE_BUILD_TYPE=Release"
        )
        self._run(cmake_configure, cwd=self.llama_dir)
        self._run(f"cmake --build build --config Release -j {build_threads}", cwd=self.llama_dir)

    def _convert_to_gguf_f16(self):
        """Convert the HuggingFace model to GGUF F16 format."""
        # Check disk space before starting a potentially long conversion
        f16_size_gb = (self.model_params_b * 1e9 * 2) / 1024**3  # FP16 = 2 bytes/param
        free_gb = shutil.disk_usage(str(self.gguf_dir)).free / 1024**3
        if free_gb < f16_size_gb * 1.1:
            raise RuntimeError(
                f"Insufficient disk space for F16 conversion.\n"
                f"  Required  : ~{f16_size_gb:.1f} GB\n"
                f"  Available : {free_gb:.1f} GB"
            )
        print(f"  Disk space: {free_gb:.1f} GB available (required: ~{f16_size_gb:.1f} GB) ✓")
        print("\n🔄 Converting HF → GGUF F16...")
        hf_model_abs = self.hf_model.resolve()
        gguf_f16_abs = self.gguf_f16.resolve()
        self._run(
            f'"{sys.executable}" convert_hf_to_gguf.py "{hf_model_abs}" '
            f'--outtype f16 --outfile "{gguf_f16_abs}"',
            cwd=self.llama_dir
        )

    def _quantize_model(self, nthreads):
        """Quantise the GGUF model."""
        print(f"\n⚙️  Quantising to {self.quant_type}...")

        quantize_bin = self._get_quantize_bin()

        if not quantize_bin.exists():
            raise FileNotFoundError(
                f"Quantise binary not found: {quantize_bin}\n"
                "  → Re-run the build step to recompile llama.cpp."
            )

        if not self.quantization_verbose:
            print("   Quantisation logs suppressed (quantization_verbose=False)")

        self._run(
            f'"{quantize_bin}" "{self.gguf_f16}" "{self.gguf_quantized}" {self.quant_type} {nthreads}',
            quiet=not self.quantization_verbose,
        )

    def _cleanup(self):
        """Delete the intermediate F16 file to reclaim disk space."""
        if self.gguf_f16.exists():
            self.gguf_f16.unlink()
            print(f"\n🗑️  Deleted: {self.gguf_f16}")

    def main_optimize(self):
        """Run the full optimisation pipeline."""
        print(f"\n{'='*80}")
        print(f"🚀 Optimising model for CPU: {self.model_name}")
        print(f"{'='*80}")

        if not self._check_model_exists():
            return False, None

        self._auto_detect_model_params()
        self._select_quant_type()
        self._setup_directories()

        # Analyse system upfront — ctx is useful even if the model already exists
        ctx = self._analyze_ram()
        self.ctx = ctx
        build_flags, build_threads, quant_threads, cores_physical, lto = self._detect_cpu_capabilities()

        launch_cmd = (
            f"llama-cli -m {self.gguf_quantized} "
            f"--ctx-size {ctx} -t {cores_physical} --mlock -p \"Your prompt\""
        )

        if self._check_quantized_exists():
            print("\n📋 To use this model:")
            print(f"   {launch_cmd}")
            return True, self.quant_type

        # Conversion pipeline
        llama_updated = self._clone_llama_cpp()
        self._build_llama_cpp(build_flags, build_threads, force=llama_updated, lto=lto)
        self._convert_to_gguf_f16()
        self._quantize_model(quant_threads)
        self._cleanup()

        print(f"\n{'='*80}")
        print(f"✅ Done! Quantised model: {self.gguf_quantized}")
        print(f"{'='*80}")
        print(f"\n📋 To use this model:")
        print(f"   {launch_cmd}")

        return True, self.quant_type
