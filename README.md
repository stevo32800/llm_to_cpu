# 🎓 llm_to_cpu - Apprendre l'Adaptation de LLM pour CPU

Apprendre **comment adapter n'importe quel modèle de langage (jusqu'à 32B)** pour le faire tourner sur **CPU pur** avec une RAM de 32GB max.

**Un projet 100% pédagogique et très guidé.** Chaque étape est expliquée, commentée et exécutable pas à pas.

---

## 🚀 Démarrage Rapide

### 1️⃣ Installer les dépendances

> ⚠️ **Important :** Pour des performances optimales (10–20× plus rapide), il faut compiler `llama-cpp-python` avec les bons flags SIMD. Choisissez la commande selon votre OS :

**Linux / macOS :**
```bash
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_AVX2=ON -DGGML_AVX=ON" \
  pip install llama-cpp-python --force-reinstall --no-cache-dir
pip install -r requirements.txt
```

**Windows (PowerShell) :**
```powershell
$env:CMAKE_ARGS = "-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_AVX2=ON -DGGML_AVX=ON"
pip install llama-cpp-python --force-reinstall --no-cache-dir
pip install -r requirements.txt
```

**Alternative rapide (wheels pré-compilés, sans compilation) :**
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
pip install -r requirements.txt
```

### 2️⃣ Installer Ollama (optionnel)
```
👉 https://ollama.com/download
```
> Ollama dispose d'un installeur natif pour Windows, macOS et Linux — pas besoin de WSL2 sur Windows.

### 3️⃣ Lancer le notebook
```bash
jupyter notebook llm_cpu_tutorial.ipynb
```

### 4️⃣ Exécuter Section 1 (setup)
Puis continuez pas à pas avec les explications...

---

## 📁 Structure

```
llm_to_cpu/
├── llm_cpu_tutorial.ipynb     ← 🔥 LE NOTEBOOK PRINCIPAL (7 sections)
├── cpu_optimizer.py           ← 🔧 Script d'optimisation automatique
├── medium_article_llm_on_cpu.md ← 📝 Article Medium associé
├── README.md                  ← Tu es ici
├── requirements.txt           ← Dépendances pip
└── models/                    ← Créé automatiquement
    ├── hf_downloads/          ← Modèles HuggingFace téléchargés
    ├── gguf/                  ← Modèles convertis GGUF
    └── benchmarks/            ← Résultats des tests
```

---

## 🎯 Les 7 Sections du Notebook

1. **🔧 Setup & Vérifications** - Préparer l'environnement + diagnostiquer le build SIMD
2. **📊 Fondamentaux** - Mémoire, formats, quantification (avec données de perplexité réelles)
3. **⬇️ Télécharger** - Récupérer un modèle HuggingFace
4. **🔄 Convertir** - HF → GGUF (étape critique)
5. **🚀 Inférence** - Générer du texte sur CPU
6. **📈 Comparer** - Benchmark des quantifications Q3/Q4/Q5/Q8
7. **🎯 Challenge** - Adapter ton propre modèle

---

## ⏱️ Durée Estimée

| Section | Durée | Prérequis |
|---------|-------|----------|
| 1-2 | 15 min | Lecture + Exécution |
| 3 | 5-15 min | Dépend connexion (téléchargement) |
| 4 | 20-30 min | Conversion + Quantification |
| 5 | 5 min | Inférence |
| 6 | 10 min | Benchmarks |
| **Total** | **1-2h** | |

---

## 💾 Prérequis Système

- **Python:** 3.8+
- **RAM:** 16GB minimum (idéalement 32GB pour gros modèles)
- **Disque:** 30GB libre
- **OS :** Windows 10/11, Linux (Ubuntu 20.04+), macOS 12+
- **Ollama (optionnel) :** https://ollama.com/download

---

## 🎓 Ce que tu Apprendras

✅ Pourquoi les LLM nécessitent beaucoup de mémoire pour l'inférence  
✅ Qu'est-ce que la quantification GGUF et son impact réel sur la qualité (perplexité mesurée)  
✅ Télécharger un modèle HuggingFace et le convertir au format GGUF  
✅ Compiler llama-cpp-python avec les bons flags SIMD pour 10–20× de gain  
✅ Choisir le bon format de quantification (Q4_K_M, Q5_K_M, Q8_0) selon ta RAM  
✅ Générer du texte et mesurer la vitesse (tok/s)  
✅ Comparer différentes quantifications vitesse vs qualité  
✅ Adapter des modèles jusqu'à 32B  

---

## 🔗 Ressources

- 🌐 **llama.cpp** - https://github.com/ggml-org/llama.cpp
- 🤗 **HuggingFace** - https://huggingface.co/models
- 📦 **llama-cpp-python** - https://github.com/abetlen/llama-cpp-python
- 📊 **Benchmarks quantification** - https://arxiv.org/abs/2601.14277

---

## ❓ Besoin d'aide?

1. **Exécute la cellule de diagnostic SIMD** (Section 1) — elle détecte si le build est optimal
2. **Vérifie les erreurs courantes** — documentées dans le notebook
3. **Recompile llama-cpp-python** avec les CMAKE_ARGS ci-dessus si les performances sont faibles

---

## 📝 Licence

Voir [LICENSE](LICENSE)

---

## 🚀 Prêt?

```bash
# 1. Installation avec SIMD (Linux)
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_AVX2=ON -DGGML_AVX=ON" \
  pip install llama-cpp-python --force-reinstall --no-cache-dir
pip install -r requirements.txt

# 2. Lancer Jupyter
jupyter notebook llm_cpu_tutorial.ipynb
```

**Bon apprentissage!** 🎓
