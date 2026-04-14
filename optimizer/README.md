# Optimisation de modèles HuggingFace

Ce dossier contient un pipeline complet pour télécharger, quantifier et tester n'importe quel modèle HuggingFace, selon votre matériel (CPU ou GPU CUDA/ROCm).

## 📁 Structure

```
optimizer/
├── main_optimizer.py          # Point d'entrée unique (pipeline complète)
├── downloading_model/
│   └── download_model.py      # Téléchargement HuggingFace
├── cpu/
│   └── cpu_optimize.py        # Quantification GGUF optimisée CPU
├── gpu/
│   └── gpu_optimize.py        # Quantification GGUF optimisée GPU (CUDA/ROCm)
└── loading_model/
    └── inference_test.py      # Chargement et test d'inférence hardware-aware
```

## 🚀 Utilisation rapide

Tout se fait via `main_optimizer.py` depuis le dossier `optimizer/` :

```bash
python3 -m main_optimizer [model_name] [output_path] [cpu|gpu] [profil] [yes|no]
```

| Argument | Défaut | Description |
|---|---|---|
| `model_name` | `Qwen/Qwen3-14B` | Nom du modèle HuggingFace |
| `output_path` | `/models/full/` | Dossier de téléchargement |
| `optimizer_type` | `gpu` | `cpu` ou `gpu` |
| `quant_profile` | `balanced` | `precision`, `balanced`, `speed` |
| `run_test` | `yes` | `yes` ou `no` |

### Profils de quantification

| Profil | Quantification | RAM | Qualité |
|---|---|---|---|
| `precision` | Q6_K | +++  | Meilleure |
| `balanced` | Q4_K_M | ++ | Bon compromis (défaut) |
| `speed` | Q3_K_M | + | Plus rapide |

### Exemples

```bash
# GPU, profil balanced (défaut), avec test
python3 -m main_optimizer mistralai/Ministral-3B-Instruct

# CPU, haute précision, sans test
python3 -m main_optimizer Qwen/Qwen3-14B /models/full/ cpu precision no

# GPU, balanced
python3 -m main_optimizer meta-llama/Llama-3.1-8B /models/full/ gpu balanced yes
```

## 🔄 Ce que fait le pipeline

### Étape 1 — Téléchargement (`model_downloader`)
- Télécharge le modèle HuggingFace en pleine précision (FP16) via `snapshot_download`
- Ignoré si le modèle est déjà présent dans `output_path`
- Destination : `/models/full/<org>/<model>/`

### Étape 2 — Optimisation

#### CPU (`cpu_optimizer`)
- Détecte les capacités SIMD du CPU (AVX, AVX2, AVX-512, AMX)
- Compile llama.cpp avec les flags adaptés
- Convertit le modèle HF en GGUF F16, puis quantifie
- Sélectionne automatiquement la quantification selon RAM + CPU si `quant_type="auto"`
- Fichier de sortie : `/models/gguf/<model>-<QUANT>.gguf`

#### GPU (`gpu_optimizer`)
- Détecte la VRAM disponible (CUDA via `nvidia-smi`, ROCm via `rocm-smi`)
- Compile llama.cpp avec `-DGGML_CUDA=ON` ou `-DGGML_HIPBLAS=ON`
- Sélectionne automatiquement la quantification selon VRAM si `quant_type="auto"`
- Calcule `n_gpu_layers` optimal (tout en VRAM si possible, sinon offload partiel)
- Fichier de sortie : `/models/gguf/<model>-<QUANT>-cuda.gguf` (suffixe `-cuda`/`-rocm` pour éviter les conflits avec les fichiers CPU)

### Étape 3 — Test d'inférence (`inference_tester`)

Voir section dédiée ci-dessous.

## 💡 Chargeur de modèle hardware-aware : `inference_tester`

La classe `inference_tester` (dans `loading_model/inference_test.py`) est recommandée pour charger et utiliser un modèle GGUF dans vos propres scripts. Elle adapte automatiquement tous les paramètres de chargement à votre hardware :

- **Auto-détection du fichier GGUF** : trouve le bon `.gguf` dans `/models/gguf/` sans avoir à préciser le nom exact
- **Contexte optimal** : calcule `n_ctx` selon la RAM disponible
- **Threads** : utilise tous les cœurs physiques
- **Flash Attention** : activée automatiquement si supportée par llama_cpp
- **KV cache quantisé** : réduit la bande passante mémoire (Q4_0 par défaut)
- **GPU layers** : passe `n_gpu_layers` directement depuis `gpu_optimizer`

### Utilisation dans un script Python

```python
import sys, os
sys.path.insert(0, "/app/scripts/optimizer")
from loading_model.inference_test import inference_tester

# Chargement CPU uniquement
tester = inference_tester(
    model_name="mistralai/Ministral-3B-Instruct",
    quant_type="auto",          # détecte le fichier GGUF automatiquement
    chat_mode=True,
    max_tokens=500,
    n_gpu_layers=0,             # 0 = CPU only
)

# Chargement GPU (tout en VRAM)
tester = inference_tester(
    model_name="mistralai/Ministral-3B-Instruct",
    quant_type="Q3_K_M-cuda",   # suffixe -cuda pour les fichiers GPU
    chat_mode=True,
    max_tokens=500,
    n_gpu_layers=-1,            # -1 = tout sur GPU, N = N couches sur GPU
    kv_quant=3,                 # 3=Q4_0 (léger/rapide), 6=Q5_0, 8=Q8_0 (précis)
)

# Lancer le test complet (chargement + inférence + métriques)
tester.run_test()

# Ou charger le modèle seul pour l'utiliser directement
model, ram_used = tester._load_model()
response = model.create_chat_completion(
    messages=[{"role": "user", "content": "Bonjour !"}],
    max_tokens=200,
)
```

### Paramètres de `inference_tester`

| Paramètre | Défaut | Description |
|---|---|---|
| `model_name` | `Qwen/Qwen3-14B` | Nom HuggingFace du modèle |
| `model_path` | `/models/gguf/` | Dossier contenant les fichiers GGUF |
| `quant_type` | `"balanced"` | Type de quantification |
| `chat_mode` | `False` | `True` = mode instruct (chat), `False` = completion brute |
| `max_tokens` | `150` | Nombre max de tokens à générer |
| `n_gpu_layers` | `0` | `0`=CPU, `-1`=tout GPU, `N`=N couches GPU |
| `kv_quant` | `6` | Quantification KV cache : `3`=Q4_0, `6`=Q5_0, `8`=Q8_0 |
| `test_prompt` | *(défaut)* | Prompt utilisé par `run_test()` |

## 📦 Fichiers GGUF générés

```
/models/gguf/
├── Ministral-3B-Instruct-Q3_K_M.gguf         # CPU
├── Ministral-3B-Instruct-Q3_K_M-cuda.gguf    # GPU CUDA
├── Ministral-3B-Instruct-Q3_K_M-rocm.gguf    # GPU ROCm
├── Qwen3-14B-Q4_K_M.gguf                     # CPU
└── Qwen3-14B-Q4_K_M-cuda.gguf                # GPU CUDA
```

Le suffixe `-cuda`/`-rocm` sur les fichiers GPU évite les conflits de nommage avec les fichiers CPU de même quantification.
