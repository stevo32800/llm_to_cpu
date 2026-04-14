"""
Pipeline complète d'optimisation de modèle : téléchargement, optimisation CPU ou GPU, test d'inférence
pour l'utiliser avec un seul script, en ligne de commande, il suffit de faire :
   python main_optimizer.py [model_name] [output_path] [optimizer_type] [quant_profile] [run_test]

Profils de quantification :
   precision  →  Q6_K    (meilleure qualité, plus de RAM)
   balanced   →  Q4_K_M  (bon compromis, défaut)
   speed      →  Q3_K_M  (plus rapide, moins de RAM, qualité réduite)

Exemples :
   python3 main_optimizer.py                                           # Qwen3-14B, balanced, test
   python3 main_optimizer.py Qwen/Qwen3-14B                           # Spécifier le modèle
   python3 main_optimizer.py meta-llama/Llama-3.1-8B /models/ cpu precision yes  # Haute précision sur CPU avec test
   python3 main_optimizer.py Qwen/Qwen3-14B /models/ cpu speed no     # Rapide, sans test
   list models : mistralai/Ministral-3-8B-Instruct-2512,  Qwen/Qwen3-4B-Instruct-2507, microsoft/Phi-4-mini-instruct
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from downloading_model.download_model import model_downloader
from cpu.cpu_optimize import cpu_optimizer
from gpu.gpu_optimize import gpu_optimizer
from loading_model.inference_test import inference_tester

from config import (OPTIMIZER_MODEL_NAME, OPTIMIZER_OUTPUT_PATH,
                    OPTIMIZER_TYPE, OPTIMIZER_QUANT_PROFILE, OPTIMIZER_RUN_TEST)


# Correspondance profil → type de quantification
QUANT_PROFILES = {
    "precision": "Q6_K",  
    "balanced":  "Q4_K_M",  
    "speed":     "Q3_K_M", 
}



if __name__ == "__main__":
    model_name     = OPTIMIZER_MODEL_NAME
    output_path    = OPTIMIZER_OUTPUT_PATH
    optimizer_type = OPTIMIZER_TYPE
    quant_profile  = OPTIMIZER_QUANT_PROFILE
    run_test       = OPTIMIZER_RUN_TEST
    # Résolution du profil en type de quantification
    quant_type = QUANT_PROFILES.get(quant_profile.lower(), quant_profile.upper())

    print(f"\n{'='*80}")
    print(f"🚀 Pipeline complète d'optimisation de modèle")
    print(f"{'='*80}")
    print(f"📋 Configuration :")
    print(f"   Modèle        : {model_name}")
    print(f"   Optimisation  : {optimizer_type.upper()}")
    print(f"   Qualité       : {quant_profile}  →  {quant_type}")
    print(f"   Test inférence: {run_test}")
    print(f"{'='*80}\n")
    
    # 1. Téléchargement du modèle
    print(f"\n🔹 ÉTAPE 1/3 : Téléchargement du modèle")
    downloader = model_downloader(model_name, output_path)
    downloader.main_download()
    
    # 2. Optimisation
    if optimizer_type == "cpu":
        print(f"\n🔹 ÉTAPE 2/3 : Optimisation CPU")
        optimizer = cpu_optimizer(model_name, input_path=output_path, quant_type=quant_type)
        success, quant_type = optimizer.main_optimize()
        
        # 3. Test d'inférence
        if success and run_test.lower() == "yes":
            print(f"\n🔹 ÉTAPE 3/3 : Test d'inférence")
            tester = inference_tester(model_name, quant_type=quant_type,
                                      test_prompt="J'ai un problème de disque comment je pourrais optimiser mon pc ? donne moi une réponse avec étape",
                                      max_tokens=1000, chat_mode=True,
                                      kv_quant={"Q6_K": 6, "Q5_K_M": 6, "Q4_K_M": 3, "Q3_K_M": 3, "Q8_0": 8}.get(quant_type, 3),
                                      ctx=optimizer.ctx)
            tester.run_test()

    elif optimizer_type == "gpu":
        print(f"\n🔹 ÉTAPE 2/3 : Optimisation GPU")
        backend = "cuda"  # "cuda" ou "rocm"
        optimizer = gpu_optimizer(model_name, input_path=output_path, quant_type=quant_type, backend=backend)
        success, quant_type, n_gpu_layers = optimizer.main_optimize()

        if success and run_test.lower() == "yes":
            print(f"\n🔹 ÉTAPE 3/3 : Test d'inférence")
            tester = inference_tester(model_name, quant_type=quant_type,
                                      test_prompt="J'ai un problème de disque comment je pourrais optimiser mon pc ? donne moi une réponse avec étape",
                                      max_tokens=1000, chat_mode=True,
                                      kv_quant={"Q6_K": 6, "Q5_K_M": 6, "Q4_K_M": 3, "Q3_K_M": 3, "Q8_0": 8}.get(quant_type, 3),
                                      n_gpu_layers=n_gpu_layers,
                                      ctx=optimizer.ctx)
            tester.run_test()
    else:
        print(f"\n✅ Téléchargement terminé. Pour optimiser :")
        print(f"   python main_optimizer.py {model_name} {output_path} cpu balanced yes")
    
    print(f"\n{'='*80}")
    print(f"✅ Pipeline terminée !")
    print(f"{'='*80}\n")