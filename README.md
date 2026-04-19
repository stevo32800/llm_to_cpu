# 🎓 llm_to_cpu - Learn Adapting LLMs for CPU

Learn **how to adapt any language model** to run on a **pure CPU** machine with up to 32GB of RAM.

---

## 🚀 Quick Start

### 1️⃣ Install dependencies

> ⚠️ **Important:** For optimal performance (10–20× faster), you should compile `llama-cpp-python` with the correct SIMD flags. Choose the command that matches your OS:

**Linux / macOS:**
```bash
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_AVX2=ON -DGGML_AVX=ON" \
  pip install llama-cpp-python --force-reinstall --no-cache-dir
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
$env:CMAKE_ARGS = "-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_AVX2=ON -DGGML_AVX=ON"
pip install llama-cpp-python --force-reinstall --no-cache-dir
pip install -r requirements.txt
```

**Quick alternative (prebuilt wheels, no compilation):**
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
pip install -r requirements.txt
```

### 2️⃣ Install Ollama (optional)

```
👉 https://ollama.com/download
```
> Ollama provides native installers for Windows, macOS and Linux — no WSL2 required on Windows.

### 3️⃣ Launch the notebook
```bash
jupyter notebook llm_cpu_tutorial.ipynb
```

### 4️⃣ Run Section 1 (setup)
Then continue step-by-step following the explanations...

---

## 📁 Project Structure

```
llm_to_cpu/
├── llm_cpu_tutorial.ipynb    
├── cpu_optimizer.py         
├── utils.py
├── README.md             
├── requirements.txt         
└── models/                  
    ├── full/        
    └── gguf/             
```
