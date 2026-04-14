"""
Script pour optimiser un modèle HuggingFace pour CPU avec llama.cpp
"""

import subprocess
import sys
from pathlib import Path
import cpuinfo
import psutil
import json
import shutil


class cpu_optimizer:
    """
    Classe pour automatiser la conversion d'un modèle Hugging Face en format GGUF optimisé pour CPU.
    
    Étapes :
    1. Clone le dépôt llama.cpp s'il n'existe pas déjà
    2. Compile llama.cpp avec les flags d'optimisation adaptés à votre CPU
    3. Convertit le modèle Hugging Face en format GGUF avec précision F16
    4. Quantifie le modèle pour un compromis qualité/vitesse/RAM
    5. Affiche un avertissement si la quantification dégrade fortement la qualité
    6. Supprime le fichier intermédiaire F16 pour économiser de l'espace
    """
    
    BITS_PER_PARAM = {
        "F16":     16,
        "Q8_0":     8,
        "Q6_K":     6.6,
        "Q5_K_M":   5.5,
        "Q4_K_M":   4.5,
        "Q3_K_M":   3.5,
        "Q2_K":     2.6,
    }
    
    def __init__(self,model_name="Qwen/Qwen3-14B",input_path="/models/full/",output_path="/models/gguf/",
                 quant_type="Q4_K_M",quantization_verbose=True):
        
        self.model_name = model_name
        self.input_path = input_path
        self.output_path = output_path
        self.quant_type = (quant_type or "Q4_K_M").upper()
        self.quantization_verbose = quantization_verbose
        
        # Chemins
        self.llama_dir = Path("/models/llama.cpp")  # Répertoire partagé via volume Docker
        model_folder = model_name.split("/", 1)[1] if "/" in model_name else model_name
        self.hf_model = Path(input_path) / model_name
        self.gguf_dir = Path(output_path)
        self.gguf_f16 = self.gguf_dir / f"{model_folder}-f16.gguf"
        self.gguf_quantized = self.gguf_dir / f"{model_folder}-{self.quant_type}.gguf"
        
        # Détection automatique des paramètres du modèle
        self.model_params_b = None
        self.number_of_layers = None
        self.number_of_heads = None
        self.head_dim = None
        self.model_max_ctx = None
        self._params_detected = False  # Flag pour ne détecter qu'une fois
    
    def _update_quantized_path(self):
        """Met à jour le chemin de sortie quantifié selon quant_type."""
        model_folder = self.model_name.split("/", 1)[1] if "/" in self.model_name else self.model_name
        self.gguf_quantized = self.gguf_dir / f"{model_folder}-{self.quant_type}.gguf"
    
    def _get_cpu_info(self):
        """Retourne les infos CPU (mise en cache — cpuinfo est lent ~2s)."""
        if not hasattr(self, '_cpu_info_cache'):
            self._cpu_info_cache = cpuinfo.get_cpu_info()
        return self._cpu_info_cache

    def _run(self, cmd, quiet=False, **kwargs):
        """Lance une commande et lève une exception si elle échoue."""
        print(f"\n>>> {cmd}")
        if quiet:
            kwargs.update({"stdout": subprocess.DEVNULL, "stderr": subprocess.STDOUT})
        subprocess.run(cmd, shell=True, check=True, **kwargs)

    def _estimate_model_ram_gb(self, quant_type=None):
        """Estime la RAM du modèle selon la quantification."""
        quant = (quant_type or self.quant_type).upper()
        bits = self.BITS_PER_PARAM[quant]
        return (self.model_params_b * 1e9 * bits) / 8 / 1024**3 * 1.1

    def _warn_quality_tradeoff(self, ram_avail_gb):
        """Affiche un warning si la quantification choisie peut dégrader la qualité."""
        if self.quant_type in {"Q3_K_M", "Q2_K"}:
            print("⚠️  Quantification agressive : qualité réduite (plus d'erreurs/hallucinations possibles).")
            print("   Recommandation : utiliser le profil 'balanced' ou 'precision' si la RAM le permet.")
        elif self.quant_type == "Q4_K_M" and self.model_params_b >= 12 and ram_avail_gb < 24:
            print("ℹ️  Compromis correct, mais un modèle plus petit donnera souvent une meilleure qualité perçue.")

    def _select_quant_type(self):
        """Valide et applique le type de quantification demandé."""
        if self.quant_type not in self.BITS_PER_PARAM:
            raise ValueError(
                f"Quantification inconnue: {self.quant_type}. "
                f"Valeurs supportées: {', '.join(self.BITS_PER_PARAM.keys())}"
            )
        self._update_quantized_path()
        ram_avail = psutil.virtual_memory().available / 1024**3
        print(f"\n🎯 Quantification : {self.quant_type}")
        self._warn_quality_tradeoff(ram_avail)
        return self.quant_type
    
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
            
            # Extraction des paramètres
            self.number_of_layers = self.number_of_layers or config.get('num_hidden_layers', 40)
            self.number_of_heads = self.number_of_heads or config.get('num_attention_heads', 8)
            # GQA: utiliser num_key_value_heads pour le KV cache (plus petit que num_attention_heads)
            self.num_kv_heads = getattr(self, 'num_kv_heads', None) or config.get('num_key_value_heads', self.number_of_heads)
            
            hidden_size = config.get('hidden_size', 5120)
            self.head_dim = self.head_dim or (hidden_size // self.number_of_heads)
            
            self.model_max_ctx = self.model_max_ctx or config.get('max_position_embeddings',
                                config.get('max_sequence_length', config.get('n_positions', 131072)))
            
            # Estimation du nombre de paramètres
            if self.model_params_b is None:
                # 1. Essayer de lire depuis le config
                num_params = config.get('num_parameters')
                if num_params:
                    self.model_params_b = num_params / 1e9
                else:
                    # 2. Essayer d'extraire depuis le nom du modèle
                    import re
                    model_size_match = re.search(r'(\d+\.?\d*)B', self.model_name, re.IGNORECASE)
                    if model_size_match:
                        self.model_params_b = float(model_size_match.group(1))
                        print(f"   Taille extraite du nom : {self.model_params_b}B")
                    else:
                        # 3. Formule améliorée avec GQA
                        vocab_size = config.get('vocab_size', 152064)
                        intermediate_size = config.get('intermediate_size', hidden_size * 4)
                        num_kv_heads = config.get('num_key_value_heads', self.number_of_heads)
                        
                        # Attention: Q, K, V + output projection
                        attn_params = hidden_size * (hidden_size + 2 * (hidden_size // self.number_of_heads) * num_kv_heads + hidden_size)

                        # MLP: gate, up, down
                        mlp_params = 3 * hidden_size * intermediate_size
                        # Layer norms
                        norm_params = 2 * hidden_size
                        
                        params_per_layer = attn_params + mlp_params + norm_params
                        embedding_params = vocab_size * hidden_size * 2  # input + output embeddings
                        total_params = (self.number_of_layers * params_per_layer + embedding_params)
                        self.model_params_b = round(total_params / 1e9)
            
            print(f"✅ Paramètres détectés :")
            print(f"   Paramètres    : ~{self.model_params_b}B")
            print(f"   Couches       : {self.number_of_layers}")
            print(f"   Têtes         : {self.number_of_heads}")
            print(f"   Head dim      : {self.head_dim}")
            print(f"   Contexte max  : {self.model_max_ctx:,} tokens ({self.model_max_ctx/1000:.0f}k)")
            
            self._params_detected = True
            
        except Exception as e:
            print(f"⚠️  Erreur lors de la lecture du config: {e}")
            print(f"   Utilisation des valeurs par défaut")
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
    
    def _analyze_ram(self):
        """Analyse la RAM disponible et calcule le contexte optimal."""
        ram = psutil.virtual_memory()
        ram_total = ram.total / 1024**3
        ram_avail = ram.available / 1024**3
        
        print(f"\n💾 RAM totale     : {ram_total:.1f} GB")
        print(f"💾 RAM disponible : {ram_avail:.1f} GB")
        
        # Calcul de la RAM du modèle
        model_ram_gb = self._estimate_model_ram_gb()
        
        ram_after_model = ram_avail - model_ram_gb
        print(f"\n📊 Modèle {self.model_params_b}B {self.quant_type}")
        print(f"  Taille estimée   : {model_ram_gb:.1f} GB")
        print(f"  RAM après modèle : ~{ram_after_model:.1f} GB")
        
        if ram_after_model < 2:
            print(f"⚠️  RAM insuffisante ! Modèle: {model_ram_gb:.1f} GB, dispo: {ram_avail:.1f} GB")
        
        # Calcul du contexte optimal
        ram_for_ctx = max(ram_after_model - 1.5, 0.5)
        # GQA: utiliser num_kv_heads (≤ num_attention_heads) pour ne pas sur-estimer le KV cache
        num_kv_heads = getattr(self, 'num_kv_heads', self.number_of_heads)
        kv_per_token = self.number_of_layers * num_kv_heads * self.head_dim * 2 * 2
        kv_gb_per_token = kv_per_token / 1024**3
        max_ctx_by_ram = int(ram_for_ctx / kv_gb_per_token)
        
        ctx = min(max_ctx_by_ram, self.model_max_ctx)
        ctx = max(ctx, 512)
        ctx = (ctx // 512) * 512
        
        kv_cache_gb = ctx * kv_gb_per_token
        print(f"  KV cache estimé  : {kv_cache_gb:.2f} GB")
        print(f"  Contexte         : {ctx} tokens ({ctx/1000:.0f}k)")
        
        return ctx
    
    def _detect_cpu_capabilities(self):
        """Détecte les capacités SIMD du CPU."""
        info = self._get_cpu_info()
        flags = info.get('flags', [])
        
        has_avx2 = 'avx2' in flags
        has_avx512 = 'avx512f' in flags
        has_avx512vnni = 'avx512vnni' in flags
        has_amx = 'amx_tile' in flags
        
        print(f"\n🖥️  CPU : {info['brand_raw']}")
        print(f"  AVX2       : {'✓' if has_avx2 else '✗'}")
        print(f"  AVX-512    : {'✓' if has_avx512 else '✗'}")
        print(f"  AVX-512VNNI: {'✓' if has_avx512vnni else '✗'}")
        print(f"  AMX        : {'✓' if has_amx else '✗'}")
        
        cores_physical = psutil.cpu_count(logical=False) or 1
        cores_logical = psutil.cpu_count(logical=True) or cores_physical
        
        print(f"\n  Cœurs physiques: {cores_physical}")
        print(f"  Cœurs logiques : {cores_logical}")
        
        # Base flags présents sur tous les builds : native optim + OpenMP
        base_flags = "-DGGML_NATIVE=ON -DGGML_OPENMP=ON"

        # Flags SIMD explicites selon les capacités détectées
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

        # Threads de compilation (fraction conservative pour ne pas saturer)
        build_threads = (max(8, int(cores_physical * 0.6)) if cores_physical >= 16 else
                        max(4, cores_physical - 2) if cores_physical >= 8 else
                        max(1, cores_physical - 1))

        # Threads de quantization : tous les cœurs logiques (tâche CPU-bound)
        quant_threads = cores_logical

        print(f"\n  Build flags   : {build_flags}")
        print(f"  Threads build : {build_threads}")
        print(f"  Threads quant : {quant_threads}")

        return build_flags, build_threads, quant_threads, cores_physical
    
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
    
    def _build_llama_cpp(self, build_flags, build_threads, force=False):
        """Compile llama.cpp avec les optimisations."""
        quantize_bin = self.llama_dir / "build/bin/llama-quantize"
        if quantize_bin.exists() and not force:
            print(f"\n✅ llama.cpp déjà compilé, compilation ignorée.")
            return
        if force and quantize_bin.exists():
            print(f"\n🔄 Recompilation après mise à jour de llama.cpp...")
        else:
            print(f"\n🔨 Compilation de llama.cpp...")
        self._run(f"cmake -B build {build_flags} -DLLAMA_LTO=ON -DCMAKE_BUILD_TYPE=Release", cwd=self.llama_dir)
        self._run(f"cmake --build build --config Release -j {build_threads}", cwd=self.llama_dir)
    
    def _convert_to_gguf_f16(self):
        """Convertit le modèle HF en GGUF F16."""
        # Vérification espace disque avant de lancer une conversion potentiellement longue
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
    
    def main_optimize(self):
        """Pipeline principal d'optimisation."""
        print(f"\n{'='*80}")
        print(f"🚀 Optimisation CPU du modèle : {self.model_name}")
        print(f"{'='*80}")
        
        # Vérifications
        if not self._check_model_exists():
            return False, None
        
        # Détection automatique des paramètres du modèle
        self._auto_detect_model_params()
        self._select_quant_type()
        
        self._setup_directories()

        # Analyse système en amont : ctx utile même si le modèle existe déjà
        ctx = self._analyze_ram()
        self.ctx = ctx
        build_flags, build_threads, quant_threads, cores_physical = self._detect_cpu_capabilities()

        launch_cmd = (
            f"llama-cli -m {self.gguf_quantized} "
            f"--ctx-size {ctx} -t {cores_physical} --mlock -p \"Votre prompt\""
        )

        if self._check_quantized_exists():
            print(f"\n📋 Pour utiliser ce modèle :")
            print(f"   {launch_cmd}")
            return True, self.quant_type
        
        # Pipeline de conversion
        llama_updated = self._clone_llama_cpp()
        self._build_llama_cpp(build_flags, build_threads, force=llama_updated)
        self._convert_to_gguf_f16()
        self._quantize_model(quant_threads)
        self._cleanup()
        
        print(f"\n{'='*80}")
        print(f"✅ Terminé ! Modèle quantisé : {self.gguf_quantized}")
        print(f"{'='*80}")
        print(f"\n📋 Pour utiliser ce modèle :")
        print(f"   {launch_cmd}")
        
        return True, self.quant_type



