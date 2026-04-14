"""
Script pour télécharger n'importe quel modèle HuggingFace en pleine précision (non quantifié)
Pour conversion GGUF avec llama.cpp ou optimisation GPU

Usage:
    python download_model_full.py Qwen/Qwen3-14B
    python download_model_full.py meta-llama/Llama-3.1-8B
    python download_model_full.py mistralai/Mistral-7B-v0.1
"""

from huggingface_hub import snapshot_download
import os
import sys



class model_downloader:
    """
    Script pour télécharger n'importe quel modèle HuggingFace en pleine précision (non quantifié)
    Pour conversion GGUF avec llama.cpp ou optimisation GPU

    Usage:
        python download_model_full.py Qwen/Qwen3-14B
        python download_model_full.py meta-llama/Llama-3.1-8B
        python download_model_full.py mistralai/Mistral-7B-v0.1
    """
    def __init__(self, model_name="Qwen/Qwen3-14B", output_path="/models/full/"):
        self.model_name = model_name
        self.output_path = output_path

    def _create_output_path(self):
        # Créer un nom de dossier à partir du nom du modèle
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

        model_folder = self.model_name.split("/")[0]  # Prendre la partie avant le premier "/"
        name_model_output = self.model_name.split("/")[1]  # Prendre la partie après le premier "/"
        OUTPUT_PATH = os.path.join(self.output_path, model_folder)  # Chemin accessible en écriture
        model_output_path = os.path.join(OUTPUT_PATH, name_model_output)  # Chemin complet du modèle
        return OUTPUT_PATH, model_output_path
    
    def _check_model_exists(self, model_output_path):
        if os.path.exists(model_output_path) and os.path.exists(f"{model_output_path}/config.json"):
            print(f"✅ Modèle déjà présent.")
            print(f"📂 Chemin: {os.path.abspath(model_output_path)}")
            print("⏭️  Téléchargement ignoré.\n")
            return True
        return False

    def _download_and_save_model(self, model_output_path):
        print(f"📥 Téléchargement du modèle: {self.model_name}")
        print(f"💾 Destination: {model_output_path}\n")
        print("⚠️  Note: Ce téléchargement peut prendre beaucoup d'espace disque\n")
        # Télécharger tous les fichiers du modèle (tokenizer + poids) directement
        print("📥 Téléchargement du modèle et tokenizer...")
        snapshot_download(
            repo_id=self.model_name,
            local_dir=model_output_path,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
        )
        print("✅ Modèle et tokenizer sauvegardés!")
        print(f"📂 Chemin complet: {os.path.abspath(model_output_path)}\n")

    def main_download(self):
        OUTPUT_PATH, model_output_path = self._create_output_path()
        if not self._check_model_exists(model_output_path):
            self._download_and_save_model(model_output_path)
        print("✨ Prêt pour l'optimisation!")

