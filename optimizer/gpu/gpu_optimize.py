"""
Script pour optimiser un modèle HuggingFace pour GPU avec llama.cpp (CUDA/ROCm)
"""

import subprocess
import sys
from pathlib import Path
import psutil
import json
import shutil


class gpu_optimizer:
    """
    Classe pour automatiser la conversion d'un modèle Hugging Face en format GGUF optimisé pour GPU.

    Étapes :
    1. Clone le dépôt llama.cpp s'il n'existe pas déjà
    2. Compile llama.cpp avec les flags CUDA ou ROCm
    3. Convertit le modèle Hugging Face en format GGUF avec précision F16
    4. Choisit automatiquement la quantification selon le modèle et la VRAM disponible
    5. Quantifie le modèle
    6. Calcule le nombre optimal de couches à offloader sur GPU (n_gpu_layers)
    7. Supprime le fichier intermédiaire F16 pour économiser de l'espace
    """

    BITS_PER_PARAM = {
        "F16":    16,
        "Q8_0":    8,
        "Q6_K":    6.6,
        "Q5_K_M":  5.5,
        "Q4_K_M":  4.5,
        "Q3_K_M":  3.5,
        "Q2_K":    2.6,
    }

    def __init__(self, model_name="Qwen/Qwen3-14B", input_path="/models/full/",
                 output_path="/models/gguf/", quant_type="auto",
                 backend="cuda", quantization_verbose=True):
        """
        Parameters
        ----------
        model_name : str
            Nom du modèle HuggingFace (ex: "Qwen/Qwen3-14B")
        input_path : str
            Répertoire contenant les modèles HuggingFace téléchargés
        output_path : str
            Répertoire de sortie pour les fichiers GGUF
        quant_type : str
            Type de quantification ("Q4_K_M", "Q6_K", etc.)
        backend : str
            Backend GPU à utiliser : "cuda" (NVIDIA) ou "rocm" (AMD)
        quantization_verbose : bool
            Afficher les logs de quantification
        """
        self.model_name = model_name
        self.input_path = input_path
        self.output_path = output_path
        self.quant_type = (quant_type or "Q4_K_M").upper()
        self.backend = backend.lower()
        self.quantization_verbose = quantization_verbose

        # Chemins
        self.llama_dir = Path("/models/llama.cpp")
        model_folder = model_name.split("/", 1)[1] if "/" in model_name else model_name
        self.hf_model = Path(input_path) / model_name
        self.gguf_dir = Path(output_path)
        self.gguf_f16 = self.gguf_dir / f"{model_folder}-f16.gguf"
        self.gguf_quantized = self.gguf_dir / f"{model_folder}-{self.quant_type}-{self.backend}.gguf"

        # Paramètres du modèle (détectés automatiquement)
        self.model_params_b = None
        self.number_of_layers = None
        self.number_of_heads = None
        self.head_dim = None
        self.model_max_ctx = None
        self._params_detected = False

        # Résultats GPU
        self.n_gpu_layers = 0
        self.vram_total_gb = 0.0
        self.vram_avail_gb = 0.0

    # ─────────────────────────────────────────────
    # Utilitaires internes
    # ─────────────────────────────────────────────

    def _update_quantized_path(self):
        """Met à jour le chemin de sortie quantifié selon quant_type."""
        model_folder = self.model_name.split("/", 1)[1] if "/" in self.model_name else self.model_name
        self.gguf_quantized = self.gguf_dir / f"{model_folder}-{self.quant_type}-{self.backend}.gguf"

    def _run(self, cmd, quiet=False, **kwargs):
        """Lance une commande shell et lève une exception si elle échoue."""
        print(f"\n>>> {cmd}")
        if quiet:
            kwargs.update({"stdout": subprocess.DEVNULL, "stderr": subprocess.STDOUT})
        subprocess.run(cmd, shell=True, check=True, **kwargs)

    def _estimate_model_ram_gb(self, quant_type=None):
        """Estime la VRAM nécessaire pour le modèle selon la quantification."""
        quant = (quant_type or self.quant_type).upper()
        bits = self.BITS_PER_PARAM.get(quant, 4.5)
        return (self.model_params_b * 1e9 * bits) / 8 / 1024**3 * 1.15  # +15% overhead GPU

    # ─────────────────────────────────────────────
    # Détection GPU
    # ─────────────────────────────────────────────

    def _detect_gpu(self):
        """Détecte le GPU disponible et retourne la VRAM (total, libre) en GB."""
        print(f"\n🖥️  Détection GPU ({self.backend.upper()})...")

        if self.backend == "cuda":
            return self._detect_cuda()
        elif self.backend == "rocm":
            return self._detect_rocm()
        else:
            raise ValueError(f"Backend inconnu : {self.backend}. Utilisez 'cuda' ou 'rocm'.")

    def _detect_cuda(self):
        """Détecte les GPUs NVIDIA via nvidia-smi."""
        try:
            result = subprocess.run(
                "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits",
                shell=True, capture_output=True, text=True, check=True
            )
            lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            if not lines:
                raise RuntimeError("Aucun GPU NVIDIA détecté.")

            total_vram = 0.0
            free_vram = 0.0
            for i, line in enumerate(lines):
                parts = [p.strip() for p in line.split(",")]
                name, mem_total_mib, mem_free_mib = parts[0], float(parts[1]), float(parts[2])
                total_gb = mem_total_mib / 1024
                free_gb = mem_free_mib / 1024
                total_vram += total_gb
                free_vram += free_gb
                print(f"   GPU {i}: {name}")
                print(f"          VRAM totale : {total_gb:.1f} GB")
                print(f"          VRAM libre  : {free_gb:.1f} GB")

            self.vram_total_gb = total_vram
            self.vram_avail_gb = free_vram
            return total_vram, free_vram

        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  nvidia-smi introuvable ou aucun GPU NVIDIA.")
            self.vram_total_gb = 0.0
            self.vram_avail_gb = 0.0
            return 0.0, 0.0

    def _detect_rocm(self):
        """Détecte les GPUs AMD via rocm-smi."""
        try:
            result = subprocess.run(
                "rocm-smi --showmeminfo vram --csv",
                shell=True, capture_output=True, text=True, check=True
            )
            total_vram = 0.0
            free_vram = 0.0
            for line in result.stdout.strip().splitlines()[1:]:
                parts = line.split(",")
                if len(parts) >= 3:
                    try:
                        used_bytes = int(parts[1].strip())
                        total_bytes = int(parts[2].strip())
                        total_gb = total_bytes / 1024**3
                        free_gb = (total_bytes - used_bytes) / 1024**3
                        total_vram += total_gb
                        free_vram += free_gb
                        print(f"   GPU AMD — VRAM totale : {total_gb:.1f} GB, libre : {free_gb:.1f} GB")
                    except ValueError:
                        pass

            self.vram_total_gb = total_vram
            self.vram_avail_gb = free_vram
            return total_vram, free_vram

        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  rocm-smi introuvable ou aucun GPU AMD.")
            self.vram_total_gb = 0.0
            self.vram_avail_gb = 0.0
            return 0.0, 0.0

    # ─────────────────────────────────────────────
    # Quantification
    # ─────────────────────────────────────────────

    def _select_quant_type(self):
        """Valide et applique le type de quantification demandé."""
        if self.quant_type not in self.BITS_PER_PARAM:
            raise ValueError(
                f"Quantification inconnue: {self.quant_type}. "
                f"Valeurs supportées: {', '.join(self.BITS_PER_PARAM.keys())}"
            )
        self._update_quantized_path()
        print(f"\n🎯 Quantification : {self.quant_type}")
        return self.quant_type

    # ─────────────────────────────────────────────
    # Paramètres du modèle
    # ─────────────────────────────────────────────

    def _auto_detect_model_params(self):
        """Détecte automatiquement les paramètres du modèle depuis config.json."""
        if self._params_detected:
            return

        config_path = self.hf_model / "config.json"

        if not config_path.exists():
            print(f"⚠️  config.json non trouvé, utilisation des valeurs par défaut")
            self._set_default_params()
            self._params_detected = True
            return

        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            print(f"\n🔍 Détection automatique des paramètres depuis config.json...")

            self.number_of_layers = self.number_of_layers or config.get('num_hidden_layers', 40)
            self.number_of_heads = self.number_of_heads or config.get('num_attention_heads', 8)
            self.num_kv_heads = getattr(self, 'num_kv_heads', None) or config.get('num_key_value_heads', self.number_of_heads)

            hidden_size = config.get('hidden_size', 5120)
            self.head_dim = self.head_dim or (hidden_size // self.number_of_heads)
            self.model_max_ctx = self.model_max_ctx or config.get(
                'max_position_embeddings',
                config.get('max_sequence_length', config.get('n_positions', 131072))
            )

            if self.model_params_b is None:
                num_params = config.get('num_parameters')
                if num_params:
                    self.model_params_b = num_params / 1e9
                else:
                    import re
                    match = re.search(r'(\d+\.?\d*)B', self.model_name, re.IGNORECASE)
                    if match:
                        self.model_params_b = float(match.group(1))
                        print(f"   Taille extraite du nom : {self.model_params_b}B")
                    else:
                        vocab_size = config.get('vocab_size', 152064)
                        intermediate_size = config.get('intermediate_size', hidden_size * 4)
                        num_kv_heads = config.get('num_key_value_heads', self.number_of_heads)
                        attn_params = hidden_size * (hidden_size + 2 * (hidden_size // self.number_of_heads) * num_kv_heads + hidden_size)
                        mlp_params = 3 * hidden_size * intermediate_size
                        norm_params = 2 * hidden_size
                        params_per_layer = attn_params + mlp_params + norm_params
                        embedding_params = vocab_size * hidden_size * 2
                        self.model_params_b = round((self.number_of_layers * params_per_layer + embedding_params) / 1e9)

            print(f"✅ Paramètres détectés :")
            print(f"   Paramètres    : ~{self.model_params_b}B")
            print(f"   Couches       : {self.number_of_layers}")
            print(f"   Têtes         : {self.number_of_heads}")
            print(f"   Head dim      : {self.head_dim}")
            print(f"   Contexte max  : {self.model_max_ctx:,} tokens ({self.model_max_ctx/1000:.0f}k)")

            self._params_detected = True

        except Exception as e:
            print(f"⚠️  Erreur lors de la lecture du config: {e}")
            self._set_default_params()
            self._params_detected = True

    def _set_default_params(self):
        """Définit les paramètres par défaut si la détection échoue."""
        self.model_params_b = self.model_params_b or 14
        self.number_of_layers = self.number_of_layers or 40
        self.number_of_heads = self.number_of_heads or 8
        self.num_kv_heads = getattr(self, 'num_kv_heads', None) or 8
        self.head_dim = self.head_dim or 128
        self.model_max_ctx = self.model_max_ctx or 131072

    # ─────────────────────────────────────────────
    # Calcul n_gpu_layers
    # ─────────────────────────────────────────────

    def _calculate_gpu_layers(self):
        """
        Calcule le nombre optimal de couches à offloader sur GPU.

        Stratégie :
        - Chaque couche pèse environ model_size / number_of_layers
        - On réserve 10% de VRAM pour le KV cache et le runtime
        - Si toutes les couches rentrent → -1 (tout sur GPU)
        """
        if self.vram_avail_gb <= 0:
            print("⚠️  Pas de VRAM disponible, inférence sur CPU uniquement (n_gpu_layers=0)")
            self.n_gpu_layers = 0
            return 0

        model_size_gb = self._estimate_model_ram_gb()
        vram_budget = self.vram_avail_gb * 0.90  # 10% réserve runtime

        if model_size_gb <= vram_budget:
            self.n_gpu_layers = -1  # Tout sur GPU
            print(f"\n✅ Modèle entier en VRAM (n_gpu_layers=-1)")
            print(f"   Modèle : {model_size_gb:.1f} GB ≤ budget VRAM : {vram_budget:.1f} GB")
        else:
            gb_per_layer = model_size_gb / self.number_of_layers
            layers = int(vram_budget / gb_per_layer)
            self.n_gpu_layers = max(0, layers)
            offload_pct = self.n_gpu_layers / self.number_of_layers * 100
            print(f"\n⚙️  Offload partiel GPU (n_gpu_layers={self.n_gpu_layers}/{self.number_of_layers} = {offload_pct:.0f}%)")
            print(f"   Modèle : {model_size_gb:.1f} GB, budget VRAM : {vram_budget:.1f} GB")
            print(f"   {self.number_of_layers - self.n_gpu_layers} couche(s) resteront sur CPU")

        return self.n_gpu_layers

    def _calculate_ctx_from_vram(self):
        """
        Calcule le contexte optimal selon la VRAM restante après le modèle.
        Utilisé pour le paramètre --ctx-size de la commande de lancement.
        """
        if self.vram_avail_gb <= 0:
            print("⚠️  Pas de VRAM détectée, contexte par défaut (4096 tokens)")
            return 4096

        model_size_gb = self._estimate_model_ram_gb()
        vram_for_ctx = max(self.vram_avail_gb - model_size_gb - 0.5, 0.5)

        num_kv_heads = getattr(self, 'num_kv_heads', self.number_of_heads)
        kv_per_token = self.number_of_layers * num_kv_heads * self.head_dim * 2 * 2
        kv_gb_per_token = kv_per_token / 1024**3
        max_ctx_by_vram = int(vram_for_ctx / kv_gb_per_token)

        ctx = min(max_ctx_by_vram, self.model_max_ctx)
        ctx = max(ctx, 512)
        ctx = (ctx // 512) * 512

        kv_cache_gb = ctx * kv_gb_per_token
        print(f"\n  KV cache estimé  : {kv_cache_gb:.2f} GB")
        print(f"  Contexte optimal : {ctx} tokens ({ctx/1000:.0f}k)")
        return ctx

    # ─────────────────────────────────────────────
    # Vérifications et setup
    # ─────────────────────────────────────────────

    def _check_model_exists(self):
        """Vérifie que le modèle source existe."""
        if not self.hf_model.exists():
            print(f"❌ Erreur : Le modèle {self.hf_model} n'existe pas")
            print(f"   Téléchargez-le d'abord avec :")
            print(f"   python main_optimizer.py {self.model_name}")
            print(f"\n   Modèles disponibles dans {self.input_path} :")
            full_dir = Path(self.input_path)
            if full_dir.exists():
                for f in full_dir.iterdir():
                    if f.is_dir():
                        print(f"   - {f.name}")
            return False
        return True

    def _check_quantized_exists(self):
        """Vérifie si le modèle quantisé existe déjà."""
        if self.gguf_quantized.exists():
            file_size = self.gguf_quantized.stat().st_size / 1024**3
            print(f"\n✅ Modèle quantisé déjà présent !")
            print(f"📂 Chemin : {self.gguf_quantized}")
            print(f"📊 Taille : {file_size:.2f} GB")
            print(f"\n⏭️  Quantization ignorée.")
            return True
        return False

    def _setup_directories(self):
        """Crée les répertoires nécessaires."""
        self.gguf_dir.mkdir(parents=True, exist_ok=True)
        print(f"📂 Modèle source : {self.hf_model}")
        print(f"📂 Sortie GGUF   : {self.gguf_dir}")
        print(f"📂 llama.cpp     : {self.llama_dir}")

    # ─────────────────────────────────────────────
    # Build llama.cpp
    # ─────────────────────────────────────────────

    def _clone_llama_cpp(self):
        """Clone ou met à jour le dépôt llama.cpp. Retourne True si des changements ont été pullés."""
        if not self.llama_dir.exists():
            print(f"\n📥 Clonage de llama.cpp dans {self.llama_dir}...")
            self.llama_dir.parent.mkdir(parents=True, exist_ok=True)
            self._run("git clone --depth=1 https://github.com/ggml-org/llama.cpp", cwd=self.llama_dir.parent)
            return True

        print(f"\n🔄 Mise à jour de llama.cpp ({self.llama_dir})...")
        result_before = subprocess.run(
            "git rev-parse HEAD", shell=True, check=True,
            capture_output=True, text=True, cwd=self.llama_dir
        ).stdout.strip()
        self._run("git fetch --depth=1 origin", cwd=self.llama_dir)
        self._run("git reset --hard FETCH_HEAD", cwd=self.llama_dir)
        result_after = subprocess.run(
            "git rev-parse HEAD", shell=True, check=True,
            capture_output=True, text=True, cwd=self.llama_dir
        ).stdout.strip()
        updated = result_before != result_after
        if updated:
            print(f"   Nouveau commit : {result_after[:8]} (ancien : {result_before[:8]})")
        else:
            print(f"   Déjà à jour ({result_after[:8]}).")
        return updated

    def _build_llama_cpp(self, build_threads, force=False):
        """Compile llama.cpp avec les flags CUDA ou ROCm."""
        quantize_bin = self.llama_dir / "build/bin/llama-quantize"
        if quantize_bin.exists() and not force:
            print(f"\n✅ llama.cpp déjà compilé, compilation ignorée.")
            return
        if force and quantize_bin.exists():
            print(f"\n🔄 Recompilation après mise à jour de llama.cpp...")
        elif self.backend == "cuda":
            print(f"\n🔨 Compilation de llama.cpp avec CUDA...")
        elif self.backend == "rocm":
            print(f"\n🔨 Compilation de llama.cpp avec ROCm/HIP...")
        else:
            print(f"\n🔨 Compilation de llama.cpp (backend inconnu, flags par défaut)...")

        if self.backend == "cuda":
            # Detect CUDA compute capability dynamically (cmake 3.22 doesn't support 'native')
            try:
                import torch
                cc = torch.cuda.get_device_capability(0)
                cuda_arch = f"{cc[0]}{cc[1]}"
            except Exception:
                cuda_arch = "native"
            build_flags = f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={cuda_arch}"
        elif self.backend == "rocm":
            build_flags = "-DGGML_HIPBLAS=ON"
        else:
            build_flags = ""

        self._run(f"cmake -B build {build_flags} -DLLAMA_LTO=ON -DCMAKE_BUILD_TYPE=Release", cwd=self.llama_dir)
        self._run(f"cmake --build build --config Release -j {build_threads}", cwd=self.llama_dir)

    # ─────────────────────────────────────────────
    # Conversion et quantification
    # ─────────────────────────────────────────────

    def _convert_to_gguf_f16(self):
        """Convertit le modèle HF en GGUF F16."""
        f16_size_gb = (self.model_params_b * 1e9 * 2) / 1024**3  # FP16 = 2 octets/param
        free_gb = shutil.disk_usage(str(self.gguf_dir)).free / 1024**3
        if free_gb < f16_size_gb * 1.1:
            raise RuntimeError(
                f"Espace disque insuffisant pour la conversion F16.\n"
                f"  Requis  : ~{f16_size_gb:.1f} GB\n"
                f"  Disponible : {free_gb:.1f} GB"
            )
        print(f"  Espace disque : {free_gb:.1f} GB disponible (requis : ~{f16_size_gb:.1f} GB) ✓")
        print(f"\n🔄 Conversion HF → GGUF F16...")
        self._run(
            f"{sys.executable} convert_hf_to_gguf.py {self.hf_model} "
            f"--outtype f16 --outfile {self.gguf_f16}",
            cwd=self.llama_dir
        )

    def _quantize_model(self, nthreads):
        """Quantifie le modèle GGUF."""
        print(f"\n⚙️  Quantization en {self.quant_type}...")

        quantize_bin = (self.llama_dir / "build/bin/Release/llama-quantize.exe" if sys.platform == "win32"
                        else self.llama_dir / "build/bin/llama-quantize")

        if not quantize_bin.exists():
            raise FileNotFoundError(f"Binaire de quantization introuvable : {quantize_bin}")

        if not self.quantization_verbose:
            print("   Logs quantization masqués (quantization_verbose=False)")
        self._run(
            f"{quantize_bin} {self.gguf_f16} {self.gguf_quantized} {self.quant_type} {nthreads}",
            quiet=not self.quantization_verbose,
        )

    def _cleanup(self):
        """Supprime le fichier F16 intermédiaire."""
        if self.gguf_f16.exists():
            self.gguf_f16.unlink()
            print(f"\n🗑️  Supprimé : {self.gguf_f16}")

    # ─────────────────────────────────────────────
    # Pipeline principal
    # ─────────────────────────────────────────────

    def main_optimize(self):
        """Pipeline principal d'optimisation GPU."""
        print(f"\n{'='*80}")
        print(f"🚀 Optimisation GPU ({self.backend.upper()}) du modèle : {self.model_name}")
        print(f"{'='*80}")

        if not self._check_model_exists():
            return False, None

        # Détection GPU
        self._detect_gpu()

        # Paramètres du modèle
        self._auto_detect_model_params()
        self._select_quant_type()

        self._setup_directories()

        # Calcul n_gpu_layers et contexte optimal (utile même si GGUF déjà présent)
        n_gpu_layers = self._calculate_gpu_layers()
        ctx = self._calculate_ctx_from_vram()
        self.ctx = ctx

        # Threads : build conservatif, quantization sur tous les cœurs logiques
        cores_physical = psutil.cpu_count(logical=False) or 1
        cores_logical = psutil.cpu_count(logical=True) or cores_physical
        build_threads = (max(8, int(cores_physical * 0.6)) if cores_physical >= 16 else
                        max(4, cores_physical - 2) if cores_physical >= 8 else
                        max(1, cores_physical - 1))
        quant_threads = cores_logical

        ngl_info = '(tout sur GPU)' if n_gpu_layers == -1 else f'({self.number_of_layers - n_gpu_layers} couche(s) sur CPU)'
        launch_cmd = (
            f"llama-cli -m {self.gguf_quantized} "
            f"-ngl {n_gpu_layers} --ctx-size {ctx} -fa --mlock -p \"Votre prompt\""
        )

        if self._check_quantized_exists():
            print(f"\n📋 Pour utiliser ce modèle avec GPU :")
            print(f"   {launch_cmd}")
            print(f"\n   n_gpu_layers = {n_gpu_layers}  {ngl_info}")
            return True, f"{self.quant_type}-{self.backend}", n_gpu_layers

        # Conversion et quantification
        llama_updated = self._clone_llama_cpp()
        self._build_llama_cpp(build_threads, force=llama_updated)
        self._convert_to_gguf_f16()
        self._quantize_model(quant_threads)
        self._cleanup()

        print(f"\n{'='*80}")
        print(f"✅ Terminé ! Modèle quantisé : {self.gguf_quantized}")
        print(f"{'='*80}")
        print(f"\n📋 Pour utiliser ce modèle avec GPU :")
        print(f"   {launch_cmd}")
        print(f"\n   n_gpu_layers = {n_gpu_layers}  {ngl_info}")

        return True, f"{self.quant_type}-{self.backend}", n_gpu_layers
