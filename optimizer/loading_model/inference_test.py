"""
Script de test d'inférence pour modèles GGUF optimisés CPU

Auto-détecte le fichier GGUF quantifié pour le modèle spécifié.

Usage:
    python inference_test.py                           # Par défaut: Qwen/Qwen3-14B (auto-détection)
    python inference_test.py Qwen/Qwen3-14B           # Spécifier le modèle (auto-détection)
    python inference_test.py meta-llama/Llama-3.1-8B  # Autre modèle (auto-détection)
"""

import sys
import subprocess
import time
from pathlib import Path
import psutil
import threading
from llama_cpp import Llama

class inference_tester:
    """Teste l'inférence d'un modèle GGUF et mesure les performances (RAM, CPU, tok/s)."""
    
    def __init__(self, model_name="Qwen/Qwen3-14B",
        model_path="/models/gguf/",
        quant_type="auto",
        test_prompt="Explique-moi ce qu'est l'intelligence artificielle en 3 phrases.",
        max_tokens=150,
        chat_mode=False,
        kv_quant=4,
        n_gpu_layers=0,
        ctx=None,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.test_prompt = test_prompt
        self.max_tokens = max_tokens
        self.chat_mode = chat_mode
        # KV cache quantization: 4=Q4_0 (fast/light), 6=Q5_0, 8=Q8_0 (accurate)
        self.kv_quant = kv_quant
        # 0 = CPU only, -1 = tout sur GPU, N = N couches sur GPU
        self.n_gpu_layers = n_gpu_layers
        
        # Auto-détection du fichier GGUF si quant_type="auto"
        model_folder = model_name.split("/", 1)[1] if "/" in model_name else model_name
        
        if quant_type.lower() == "auto":
            self.quant_type, self.gguf_file = self._auto_detect_gguf(model_folder)
        else:
            self.quant_type = quant_type
            self.gguf_file = Path(model_path) / f"{model_folder}-{quant_type}.gguf"
        
        # Info système
        self.cores_physical = psutil.cpu_count(logical=False) or 1
        self.cores_logical = psutil.cpu_count(logical=True) or self.cores_physical
        
        # Contexte : utilise la valeur passée par l'optimiseur si disponible, sinon auto-calcul
        self.ctx = ctx if ctx is not None else self._calculate_optimal_context()
        self.cpu_usage = []
        self.vram_usage = []
        self.monitoring = False
    
    def _auto_detect_gguf(self, model_folder):
        """Détecte automatiquement le fichier GGUF existant pour ce modèle."""
        gguf_dir = Path(self.model_path)
        
        if not gguf_dir.exists():
            print(f"⚠️  Dossier {gguf_dir} introuvable, utilisation Q4_K_M par défaut")
            return "Q4_K_M", gguf_dir / f"{model_folder}-Q4_K_M.gguf"
        
        # Chercher tous les fichiers GGUF pour ce modèle
        pattern = f"{model_folder}-*.gguf"
        gguf_files = list(gguf_dir.glob(pattern))
        
        if not gguf_files:
            print(f"⚠️  Aucun fichier GGUF trouvé pour {model_folder}, utilisation Q4_K_M par défaut")
            return "Q4_K_M", gguf_dir / f"{model_folder}-Q4_K_M.gguf"
        
        # Prendre le premier fichier trouvé (excluant f16)
        gguf_files_filtered = [f for f in gguf_files if not f.stem.endswith('-f16')]
        
        if not gguf_files_filtered:
            gguf_files_filtered = gguf_files
        
        selected_file = gguf_files_filtered[0]
        
        # Extraire le quant_type du nom de fichier
        # Format: model_folder-QUANT_TYPE.gguf
        quant_type = selected_file.stem.replace(f"{model_folder}-", "")
        
        print(f"✅ Fichier GGUF détecté : {selected_file.name}")
        print(f"   Quantization      : {quant_type}")
        
        return quant_type, selected_file
    
    def _calculate_optimal_context(self):
        """Calcule le contexte optimal.

        - GPU : interroge la VRAM libre pour estimer un contexte réaliste.
        - CPU : basé sur la RAM disponible.
        Règle : contexte petit = KV cache léger = inférence plus rapide.
        """
        if self.n_gpu_layers != 0:
            try:
                result = subprocess.run(
                    "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits",
                    shell=True, capture_output=True, text=True, timeout=5, check=True
                )
                vram_free_mb = sum(float(l.strip()) for l in result.stdout.strip().splitlines() if l.strip())
                vram_for_ctx = max(vram_free_mb / 1024 - 0.5, 0.5)
                ctx = int(vram_for_ctx * 512)  # ~2 MB/token (estimation conservative)
                ctx = max(ctx, 512)
                ctx = min(ctx, 131072)
                return (ctx // 512) * 512
            except Exception:
                return 4096  # fallback sans nvidia-smi
        # CPU : basé sur la RAM disponible
        ram_avail = psutil.virtual_memory().available / 1024**3
        return (4096 if ram_avail > 50 else
                4096 if ram_avail > 30 else
                2048 if ram_avail > 20 else
                1024)
    
    def _monitor_cpu(self):
        """Surveille l'utilisation CPU en arrière-plan."""
        while self.monitoring:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_usage.append(cpu_percent)
            time.sleep(0.1)

    def _monitor_vram(self):
        """Surveille la VRAM NVIDIA en arrière-plan (via nvidia-smi)."""
        while self.monitoring:
            try:
                result = subprocess.run(
                    "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
                    shell=True, capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    vals = [float(l.strip()) for l in result.stdout.strip().splitlines() if l.strip()]
                    self.vram_usage.append(sum(vals))
            except Exception:
                pass
            time.sleep(0.5)
    
    def _check_model_exists(self):
        """Vérifie que le modèle GGUF existe."""
        if not self.gguf_file.exists():
            print(f"❌ Erreur : Le modèle {self.gguf_file} n'existe pas")
            print(f"   Optimisez-le d'abord avec :")
            print(f"   python main_optimizer.py {self.model_name}")
            return False
        return True
    
    def _load_model(self):
        """Charge le modèle GGUF avec les paramètres optimaux."""
        print(f"\n{'='*80}")
        print(f"📊 Test d'inférence : {self.model_name}")
        print(f"{'='*80}")
        print(f"\n📂 Modèle : {self.gguf_file}")
        print(f"�️  CPU   : {self.cores_physical} cœurs physiques, {self.cores_logical} logiques")
        
        # RAM avant chargement
        ram_before = psutil.virtual_memory()
        ram_before_gb = ram_before.used / 1024**3
        print(f"\n💾 RAM avant chargement : {ram_before_gb:.1f} GB / {ram_before.total/1024**3:.1f} GB")
        print(f"\n⏳ Chargement du modèle...")
        
        start_load = time.time()

        on_gpu = self.n_gpu_layers != 0
        n_threads = self.cores_physical
        # GPU : n_threads_batch sur cœurs logiques pour le prefill des couches CPU-side
        n_threads_batch = self.cores_logical if on_gpu else self.cores_physical
        # GPU : plus grand batch = meilleur débit GPU ; CPU : 512 pour ne pas saturer la RAM
        n_batch = 2048 if on_gpu else 512

        _base_kwargs = dict(
            model_path=str(self.gguf_file),
            n_threads=n_threads,
            n_threads_batch=n_threads_batch,
            n_ctx=self.ctx,
            n_batch=n_batch,
            use_mmap=True,
            use_mlock=not on_gpu,  # mlock inutile si modèle en VRAM
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )
        # KV cache quantisé → moins de bande passante mémoire par token généré
        # GGML_TYPE_Q4_0=2 (rapide, scale seul), Q4_1=3 (+ offset → plus lent), Q5_0=6, Q8_0=8
        flash_attn_active = False
        actual_kv_type = None
        try:
            model = Llama(**_base_kwargs, flash_attn=True, type_k=2, type_v=2)
            flash_attn_active = True
            actual_kv_type = 2
        except (TypeError, ValueError):
            try:
                model = Llama(**_base_kwargs, flash_attn=True, type_k=3, type_v=3)
                flash_attn_active = True
                actual_kv_type = 3
            except (TypeError, ValueError):
                model = Llama(**_base_kwargs)

        load_time = time.time() - start_load
        ram_after_gb = psutil.virtual_memory().used / 1024**3
        ram_model = ram_after_gb - ram_before_gb
        
        print(f"✅ Modèle chargé en {load_time:.2f}s")
        print(f"💾 RAM après chargement : {ram_after_gb:.1f} GB (+{ram_model:.1f} GB)")
        print(f"⚙️  Threads : {n_threads} (génération) / {n_threads_batch} (batch)")
        print(f"📝 Contexte : {self.ctx:,} tokens ({self.ctx/1000:.0f}k)")
        print(f"📦 Batch size : {n_batch}")
        kv_label = {1: "F16", 2: "Q4_0(fast)", 3: "Q4_1", 6: "Q5_0", 8: "Q8_0"}.get(actual_kv_type, "aucun")
        print(f"⚡ Flash Attention : {'✅ activé (KV ' + kv_label + ')' if flash_attn_active else '❌ non supporté par cette version llama_cpp'}")
        gpu_label = "tout" if self.n_gpu_layers == -1 else (f"{self.n_gpu_layers} couches" if self.n_gpu_layers > 0 else "aucune (CPU only)")
        print(f"🖥️  GPU layers     : {gpu_label} (n_gpu_layers={self.n_gpu_layers})")
        print(f"💬 Mode           : {'Chat (instruct)' if self.chat_mode else 'Raw completion (plus rapide)'}")
        
        return model, ram_model
    
    def _run_inference(self, model):
        """Exécute un test d'inférence et mesure les performances."""
        print(f"\n{'─'*80}")
        print(f"🧪 Test d'inférence")
        print(f"{'─'*80}")
        print(f"\n📝 Prompt : {self.test_prompt}")
        print(f"🎯 Max tokens : {self.max_tokens}")
        
        # Démarrer le monitoring ressources
        self.cpu_usage = []
        self.vram_usage = []
        self.monitoring = True
        cpu_thread = threading.Thread(target=self._monitor_cpu, daemon=True)
        cpu_thread.start()
        if self.n_gpu_layers != 0:
            threading.Thread(target=self._monitor_vram, daemon=True).start()
        
        # Mesurer le temps total
        start_total = time.time()
        
        # Lancer l'inférence
        print(f"\n⏳ Génération en cours...")
        if self.chat_mode:
            messages = [
                {"role": "system", "content": "Tu es un assistant utile. Réponds de façon concise."},
                {"role": "user", "content": self.test_prompt},
            ]
            # Stop tokens couvrant Qwen (<|im_end|>), Mistral (</s>), LLaMA (<|eot_id|>)
            stop_tokens = ["<|im_end|>", "</s>", "<|eot_id|>", "<|end|>", "<|endoftext|>"]
            response = model.create_chat_completion(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.3,
                top_p=0.3,
                repeat_penalty=1.1,
                stop=stop_tokens,
            )
            text = response['choices'][0]['message']['content']
        else:
            stop_tokens = ["<|im_end|>", "</s>", "<|eot_id|>", "<|end|>", "<|endoftext|>"]
            response = model(
                self.test_prompt,
                max_tokens=self.max_tokens,
                temperature=0.3,
                top_p=0.3,
                repeat_penalty=1.15,
                stop=stop_tokens,
            )
            text = response['choices'][0]['text']

        # Arrêter le monitoring et calculer métriques
        self.monitoring = False
        total_time = time.time() - start_total

        usage = response['usage']
        prompt_tokens = usage['prompt_tokens']
        completion_tokens = usage['completion_tokens']

        # Estimer le temps de prefill (traitement du prompt) vs génération
        # prefill est proportionnel au nombre de tokens du prompt
        # en pratique ~10-20% du temps total par token de prompt vs génération
        total_tokens_processed = prompt_tokens + completion_tokens
        prefill_ratio = prompt_tokens / total_tokens_processed if total_tokens_processed > 0 else 0
        prefill_time = total_time * prefill_ratio * 0.3  # prefill est plus rapide que la génération
        generation_time = total_time - prefill_time
        tokens_per_sec = completion_tokens / generation_time if generation_time > 0 else 0
        ram_inference_gb = psutil.virtual_memory().used / 1024**3
        avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
        max_cpu = max(self.cpu_usage) if self.cpu_usage else 0
        
        return {
            'text': text,
            'prompt_tokens': usage['prompt_tokens'],
            'completion_tokens': usage['completion_tokens'],
            'total_tokens': usage['total_tokens'],
            'total_time': total_time,
            'prefill_time': prefill_time,
            'generation_time': generation_time,
            'tokens_per_sec': tokens_per_sec,
            'ram_gb': ram_inference_gb,
            'avg_cpu': avg_cpu,
            'max_cpu': max_cpu,
            'max_vram_gb': max(self.vram_usage) / 1024 if self.vram_usage else 0.0,
        }
    
    def _display_results(self, metrics, ram_model):
        """Affiche les résultats du test."""
        print(f"\n{'='*80}")
        print(f"📈 Résultats")
        print(f"{'='*80}")
        
        print(f"\n📊 Tokens :")
        print(f"   Prompt      : {metrics['prompt_tokens']} tokens")
        print(f"   Génération  : {metrics['completion_tokens']} tokens")
        print(f"   Total       : {metrics['total_tokens']} tokens")
        
        print(f"\n⏱️  Performance :")
        print(f"   Temps total    : {metrics['total_time']:.2f}s")
        print(f"   dont prefill   : ~{metrics['prefill_time']:.2f}s (traitement du prompt)")
        print(f"   dont génération: ~{metrics['generation_time']:.2f}s")
        print(f"   Vitesse        : {metrics['tokens_per_sec']:.1f} tokens/s (génération seule)")
        
        print(f"\n💾 Ressources :")
        print(f"   RAM modèle  : {ram_model:.1f} GB")
        print(f"   RAM totale  : {metrics['ram_gb']:.1f} GB")
        print(f"   CPU moyen   : {metrics['avg_cpu']:.1f}%")
        print(f"   CPU max     : {metrics['max_cpu']:.1f}%")
        if metrics.get('max_vram_gb', 0) > 0:
            print(f"   VRAM GPU max: {metrics['max_vram_gb']:.1f} GB")
        
        print(f"\n💬 Réponse générée :")
        print(f"{'─'*80}")
        print(metrics['text'])
        print(f"{'─'*80}")
        
        # Benchmark qualitatif (seuils adaptés CPU/GPU)
        tok_s = metrics['tokens_per_sec']
        if self.n_gpu_layers != 0:
            benchmark = ("\u2705 Excellent" if tok_s > 60 else
                        "\u2705 Bon" if tok_s > 30 else
                        "⚠️  Moyen" if tok_s > 15 else
                        "❌ Lent")
        else:
            benchmark = ("\u2705 Excellent" if tok_s > 20 else
                        "\u2705 Bon" if tok_s > 10 else
                        "⚠️  Moyen" if tok_s > 5 else
                        "❌ Lent")
        print(f"\n🎯 Benchmark :")
        print(f"   {benchmark} ({tok_s:.1f} tok/s)")
        
        print(f"\n{'='*80}")
    
    def run_test(self):
        """Pipeline principal de test."""
        if not self._check_model_exists():
            return False
        
        # Charger le modèle
        model, ram_model = self._load_model()
        
        # Exécuter l'inférence
        metrics = self._run_inference(model)
        
        # Afficher les résultats
        self._display_results(metrics, ram_model)
        
        return True
