"""
API ultra-légère pour la prédiction de tweets de catastrophe.
Utilise Hugging Face Inference API au lieu de charger le modèle localement.
Cela permet de fonctionner sur Render (512 Mo) sans crash mémoire.
"""

import os
import re
from functools import lru_cache
from typing import Dict, List, Optional

import emoji
import requests
from fastapi import FastAPI
from pydantic import BaseModel

# --- CONFIGURATION ---
# Ces variables DOIVENT être définies dans Render (Environment Variables)
HF_API_URL = os.getenv(
    "HF_API_URL",
    "https://api-inference.huggingface.co/models/Oscarkaf/disaster-tweets-bert",
)
HF_TOKEN = os.getenv("HF_TOKEN", "")

app = FastAPI(
    title="Disaster Tweet BERT API (Hugging Face)",
    description="API ultra-légère utilisant Hugging Face Inference API.",
    version="2.0.0",
)

# --- TRANSLATION UTILS (LRU CACHE) ---
try:
    from deep_translator import GoogleTranslator
    from langdetect import detect
except ImportError:
    detect = None
    GoogleTranslator = None


@lru_cache(maxsize=128)
def translate_text(text: str) -> Dict[str, str]:
    """Détecte la langue et traduit en anglais si nécessaire. Cache LRU."""
    res = {"translated_text": text, "detected_lang": "en", "is_translated": False}
    if not text or not detect or not GoogleTranslator:
        return res
    try:
        lang = detect(text)
        res["detected_lang"] = lang
        if lang not in ["en", "eng"]:
            translated = GoogleTranslator(source=lang, target="en").translate(text)
            if translated:
                res["translated_text"] = translated
                res["is_translated"] = True
    except Exception:
        pass
    return res


# --- TEXT CLEANING ---


def clean_text_advanced(text: str) -> str:
    """Nettoie un tweet : supprime URLs, mentions, emojis, caractères non-ASCII."""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+", "[USER]", text)
    text = emoji.demojize(text)
    text = text.replace(":", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.encode("ascii", "ignore").decode("ascii")


# --- SCHEMAS ---


class TweetInput(BaseModel):
    text: str
    location: Optional[str] = None
    keyword: Optional[str] = None


class PredictionOutput(BaseModel):
    is_disaster: bool
    confidence: float
    clean_text: str
    model_name: str
    impact_words: Dict[str, float]
    detected_lang: str
    translated_text: str


# --- CORE LOGIC ---


def query_huggingface(text: str) -> tuple:
    """Appelle l'API Hugging Face Inference pour obtenir une prédiction."""
    if not HF_TOKEN:
        return None, "HF_TOKEN manquant dans les variables d'environnement Render."

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": text, "options": {"wait_for_model": True}}

    try:
        response = requests.post(
            HF_API_URL, headers=headers, json=payload, timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                preds = result[0] if isinstance(result[0], list) else result
                # Cherche le label positif (catastrophe)
                for pred in preds:
                    label = str(pred.get("label", "")).upper()
                    if label in ["LABEL_1", "POSITIVE", "DISASTER", "1"]:
                        return float(pred["score"]), None
                # Si le label 0 est trouvé en premier, inverser le score
                for pred in preds:
                    label = str(pred.get("label", "")).upper()
                    if label in ["LABEL_0", "NEGATIVE", "NOT_DISASTER", "0"]:
                        return 1.0 - float(pred["score"]), None
                # Fallback : retourne le premier score
                return float(preds[0]["score"]), None
        return None, f"Erreur HF {response.status_code}: {response.text[:200]}"
    except Exception as exc:
        return None, str(exc)


def query_huggingface_batch(texts: List[str]) -> List[float]:
    """Appelle HF en batch pour plusieurs textes (utilisé pour l'explicabilité)."""
    if not HF_TOKEN or not texts:
        return [0.0] * len(texts)

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": texts, "options": {"wait_for_model": True}}

    try:
        response = requests.post(
            HF_API_URL, headers=headers, json=payload, timeout=60
        )
        if response.status_code == 200:
            results = response.json()
            confidences = []
            for result in results:
                preds = result if isinstance(result, list) else [result]
                conf = 0.0
                for pred in preds:
                    label = str(pred.get("label", "")).upper()
                    if label in ["LABEL_1", "POSITIVE", "DISASTER", "1"]:
                        conf = float(pred["score"])
                        break
                    if label in ["LABEL_0", "NEGATIVE", "NOT_DISASTER", "0"]:
                        conf = 1.0 - float(pred["score"])
                        break
                confidences.append(conf)
            return confidences
    except Exception:
        pass

    # Fallback : appels séquentiels si le batch échoue
    results = []
    for text in texts:
        conf, err = query_huggingface(text)
        results.append(conf if conf is not None else 0.0)
    return results


def heuristic_prediction(text: str) -> float:
    """Prédiction heuristique de secours si HF est indisponible."""
    disaster_terms = {
        "earthquake",
        "flood",
        "wildfire",
        "fire",
        "hurricane",
        "evacuation",
        "disaster",
        "collapsed",
        "injured",
        "dead",
        "tsunami",
        "explosion",
        "rescue",
        "storm",
        "collision",
        "crash",
        "emergency",
        "alert",
    }
    words = set(re.findall(r"\w+", text.lower()))
    matches = words.intersection(disaster_terms)
    if matches:
        return min(0.9, 0.4 + (len(matches) * 0.15))
    return 0.15


def explain_prediction(text: str, base_confidence: float) -> Dict[str, float]:
    """Calcule l'importance de chaque mot par ablation (occlusion)."""
    words = text.split()
    if not words:
        return {}

    # Limiter à 10 mots pour ne pas surcharger l'API HF
    words_to_test = words[:10]

    # Créer les variantes (texte sans chaque mot)
    variations = []
    for i in range(len(words_to_test)):
        ablated = " ".join(words_to_test[:i] + words_to_test[i + 1 :])
        variations.append(ablated if ablated.strip() else "[EMPTY]")

    # Appel batch à HF
    ablated_confidences = query_huggingface_batch(variations)

    impacts = {}
    for i, word in enumerate(words_to_test):
        conf = (
            ablated_confidences[i] if i < len(ablated_confidences) else base_confidence
        )
        impacts[word] = round(base_confidence - conf, 4)

    return dict(sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True))


# --- ENDPOINTS ---


@app.get("/")
def home():
    """Page d'accueil de l'API."""
    return {"message": "API BERT (via Hugging Face) v2.0.0 active. /docs pour tester."}


@app.get("/health")
def health():
    """Vérifie que l'API est en ligne."""
    return {
        "status": "ok",
        "mode": "huggingface_inference",
        "model_loaded": True,
        "model_name": "BERTweet (via Hugging Face API)",
        "model_error": None,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict_tweet(tweet: TweetInput):
    """Prédit si un tweet est lié à une catastrophe."""
    # 1. Traduction
    trans_res = translate_text(tweet.text)
    work_text = trans_res["translated_text"]

    # 2. Nettoyage
    cleaned_text = clean_text_advanced(work_text)

    # 3. Texte vide → réponse immédiate
    if not cleaned_text:
        return PredictionOutput(
            is_disaster=False,
            confidence=0.0,
            clean_text="",
            model_name="N/A (texte vide)",
            impact_words={},
            detected_lang=trans_res["detected_lang"],
            translated_text=work_text,
        )

    # 4. Prédiction via HF ou fallback heuristique
    confidence, error = query_huggingface(cleaned_text)

    if error:
        confidence = heuristic_prediction(cleaned_text)
        model_used = f"Heuristic Fallback ({error[:80]})"
        impact_words: Dict[str, float] = {}
    else:
        model_used = "BERTweet (via Hugging Face API)"
        # 5. Explicabilité (importance des mots)
        impact_words = explain_prediction(cleaned_text, confidence)

    return PredictionOutput(
        is_disaster=confidence >= 0.5,
        confidence=confidence,
        clean_text=cleaned_text,
        model_name=model_used,
        impact_words=impact_words,
        detected_lang=trans_res["detected_lang"],
        translated_text=work_text,
    )
