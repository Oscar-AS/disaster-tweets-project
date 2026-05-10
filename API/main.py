"""
API ultra-légère pour la prédiction de tweets de catastrophe.
Utilise Hugging Face Inference API au lieu de charger le modèle localement.
Cela permet de fonctionner sur Render (512 Mo) sans crash mémoire.
"""

import os
import re
from functools import lru_cache
from typing import Dict, List, Optional

import json
import urllib.error
import urllib.request

import emoji
from fastapi import FastAPI
from pydantic import BaseModel

# --- CONFIGURATION ---
# Charge le fichier .env si présent (développement local)
# On cherche le .env dans le même dossier que ce fichier main.py
# pour que ça fonctionne peu importe depuis où uvicorn est lancé.
try:
    from pathlib import Path
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass  # python-dotenv non installé, on utilise les vars d'env système

# Ces variables DOIVENT être définies dans Render (Environment Variables)
# ou dans le fichier .env pour le développement local
HF_API_URL = os.getenv(
    "HF_API_URL",
    "https://api-inference.huggingface.co/models/Oscarkaf/disaster-tweets-bert"
)
HF_MODEL_ID = os.getenv("HF_MODEL_ID", HF_API_URL.rstrip("/").split("/models/")[-1])
HF_PROVIDER = os.getenv("HF_PROVIDER", "hf-inference")
HF_TOKEN = os.getenv("HF_TOKEN", "")

app = FastAPI(
    title="Disaster Tweet BERT API (Hugging Face)",
    description="API ultra-légère utilisant Hugging Face Inference API.",
    version="2.0.0",
)

# --- HF MODEL ID NORMALIZATION ---
def normalize_hf_model_id(value: str) -> str:
    """
    Rend l'identifiant compatible `InferenceClient`.
    Accepte soit:
    - "owner/repo"
    - "https://api-inference.huggingface.co/models/owner/repo"
    """
    v = (value or "").strip()
    if not v:
        return v
    marker = "/models/"
    if marker in v:
        v = v.split(marker, 1)[-1].strip().strip("/")
    return v


HF_MODEL_ID = normalize_hf_model_id(HF_MODEL_ID)

# On garde l'URL HF explicite (utile en fallback direct HTTP).
HF_API_URL = (HF_API_URL or "").strip()

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
    if not HF_MODEL_ID:
        return None, "HF_MODEL_ID manquant ou invalide."
    if not HF_API_URL:
        return None, "HF_API_URL manquant ou invalide."

    # Appel direct HTTP pour éviter toute ambiguïté de construction d'URL.
    # Hugging Face Inference API: POST { "inputs": "<text>" } sur /models/<repo_id>
    try:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        # Si le modèle est privé, HF_TOKEN est requis. Si public, on peut appeler sans.
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"
        req = urllib.request.Request(
            HF_API_URL,
            data=json.dumps({"inputs": text}).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        payload = json.loads(body) if body else None

        # Erreurs HF possibles: {"error": "..."} ou {"estimated_time": ...}
        if isinstance(payload, dict) and payload.get("error"):
            return None, str(payload.get("error"))

        # text-classification renvoie souvent: [{"label": "...", "score": ...}, ...]
        if payload:
            return extract_disaster_confidence(payload), None
    except urllib.error.HTTPError as exc:
        # Lire corps d'erreur pour message utile (401/403/404/503 etc.)
        try:
            err_body = exc.read().decode("utf-8", errors="replace")
            return None, f"{exc.code} {exc.reason}: {err_body[:400]}"
        except Exception:
            return None, f"{exc.code} {exc.reason}"
    except Exception as exc:
        return None, str(exc)

    return None, "Réponse Hugging Face vide."


def get_prediction_value(pred, key: str):
    """Lit une prédiction HF, que ce soit un dict ou un objet dataclass."""
    if isinstance(pred, dict):
        return pred.get(key)
    return getattr(pred, key, None)


def extract_disaster_confidence(preds) -> float:
    """Extrait la probabilité catastrophe depuis les labels Hugging Face."""
    positive_labels = {"LABEL_1", "POSITIVE", "DISASTER", "1"}
    negative_labels = {"LABEL_0", "NEGATIVE", "NOT_DISASTER", "0"}

    for pred in preds:
        label = str(get_prediction_value(pred, "label") or "").upper()
        if label in positive_labels:
            return float(get_prediction_value(pred, "score"))

    for pred in preds:
        label = str(get_prediction_value(pred, "label") or "").upper()
        if label in negative_labels:
            return 1.0 - float(get_prediction_value(pred, "score"))

    return float(get_prediction_value(preds[0], "score"))


def query_huggingface_batch(texts: List[str]) -> List[float]:
    """Appelle HF en batch pour plusieurs textes (utilisé pour l'explicabilité)."""
    if not HF_TOKEN or not texts:
        return [0.0] * len(texts)
    if not HF_MODEL_ID:
        return [0.0] * len(texts)
    if not HF_API_URL:
        return [0.0] * len(texts)

    results: List[float] = []
    for text in texts:
        conf, err = query_huggingface(text)
        results.append(conf if conf is not None and err is None else 0.0)
    return results

    # Fallback : appels séquentiels si le batch échoue
    # (déjà couvert par l'implémentation ci-dessus)


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
    try:
        import huggingface_hub  # type: ignore
        hub_version = getattr(huggingface_hub, "__version__", None)
    except Exception:
        hub_version = None
    return {
        "status": "ok",
        "mode": "huggingface_inference",
        "model_loaded": True,
        "model_name": "BERTweet (via Hugging Face API)",
        "model_error": None,
        "hf_model_id": HF_MODEL_ID,
        "hf_api_url": HF_API_URL,
        "huggingface_hub_version": hub_version,
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
