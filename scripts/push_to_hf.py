"""
Script pour exporter le meilleur modèle de MLflow vers Hugging Face Hub.
Usage: python scripts/push_to_hf.py
"""
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- CONFIGURATION ---
LOCAL_MODEL_PATH = "model_avancé/best_model_BERTweet"
HF_TOKEN = " " 
HF_REPO_ID = "Oscarkaf/disaster-tweets-bert"

def export():
    print(f"1. Vérification du dossier local : {LOCAL_MODEL_PATH}")
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"❌ Erreur : Le dossier {LOCAL_MODEL_PATH} n'existe pas.")
        return

    try:
        print(f"2. Chargement du modèle depuis {LOCAL_MODEL_PATH} (cela peut prendre un moment)...")
        model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        
        print(f"3. Envoi vers Hugging Face Hub: {HF_REPO_ID}...")
        # On passe le token directement ici pour éviter l'erreur de login
        model.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
        tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
        
        print("\n✅ Succès ! Votre modèle local est maintenant sur Hugging Face.")
        print(f"Lien : https://huggingface.co/{HF_REPO_ID}")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        print("\nConseil : Vérifiez votre connexion internet ou réessayez dans quelques minutes.")

if __name__ == "__main__":
    export()
