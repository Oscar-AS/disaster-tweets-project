"""
API ultra-légère pour la prédiction de tweets de catastrophe.
Charge le modèle localement via transformers.
Idéal pour Hugging Face Spaces (qui offre ~16Go de RAM).
"""

import os
import re
from functools import lru_cache
from typing import Dict, List, Optional

import json

import emoji
from fastapi import FastAPI
from pydantic import BaseModel

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

# --- CONFIGURATION ---
try:
    from pathlib import Path
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "Oscarkaf/disaster-tweets-bert")

app = FastAPI(
    title="Disaster Tweet BERT API (Local HF Space)",
    description="API chargeant le modèle localement avec transformers.",
    version="3.0.0",
)

# --- MODEL LOADING ---
_classifier_pipeline = None
_model_load_error = None

def get_classifier():
    """Charge paresseusement le modèle avec transformers."""
    global _classifier_pipeline, _model_load_error
    if _classifier_pipeline is not None:
        return _classifier_pipeline
    
    if pipeline is None:
        _model_load_error = "La bibliothèque transformers n'est pas installée."
        return None

    try:
        # On charge le modèle depuis le Hub Hugging Face (il sera mis en cache localement)
        print(f"Loading model {HF_MODEL_ID} locally...")
        _classifier_pipeline = pipeline("text-classification", model=HF_MODEL_ID)
        return _classifier_pipeline
    except Exception as e:
        _model_load_error = str(e)
        print(f"Error loading model: {_model_load_error}")
        return None

# --- TRANSLATION UTILS (LRU CACHE) ---
try:
    from deep_translator import GoogleTranslator
    from langdetect import detect
except ImportError:
    detect = None
    GoogleTranslator = None


@lru_cache(maxsize=128)
def translate_text(text: str) -> Dict[str, str]:
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

def extract_disaster_confidence_from_pipeline(preds) -> float:
    """Extrait la probabilité de catastrophe à partir des résultats du pipeline."""
    # Pipeline text-classification retourne souvent [{'label': '...', 'score': 0.9}]
    # Si le modèle est multi-label, ça peut être une liste plus complexe.
    # On va standardiser
    positive_labels = {"LABEL_1", "POSITIVE", "DISASTER", "1"}
    negative_labels = {"LABEL_0", "NEGATIVE", "NOT_DISASTER", "0"}

    # Cas d'un seul élément
    if isinstance(preds, list) and len(preds) > 0 and isinstance(preds[0], dict):
        label = str(preds[0].get("label", "")).upper()
        score = float(preds[0].get("score", 0.0))
        
        if label in positive_labels:
            return score
        elif label in negative_labels:
            return 1.0 - score
        else:
            # Si le label n'est pas reconnu, on retourne le score brut
            return score
            
    return 0.5


def query_model(text: str) -> tuple:
    """Utilise le modèle local pour obtenir une prédiction."""
    classifier = get_classifier()
    if not classifier:
        return None, _model_load_error or "Modèle non initialisé."
        
    try:
        preds = classifier(text)
        return extract_disaster_confidence_from_pipeline(preds), None
    except Exception as exc:
        return None, str(exc)


def query_model_batch(texts: List[str]) -> List[float]:
    """Prédit sur un batch de textes."""
    if not texts:
        return []
        
    classifier = get_classifier()
    if not classifier:
        return [0.0] * len(texts)
        
    try:
        preds_list = classifier(texts)
        # preds_list est une liste de [{'label': ..., 'score': ...}]
        return [extract_disaster_confidence_from_pipeline([p]) for p in preds_list]
    except Exception:
        # Fallback to sequential
        results = []
        for t in texts:
            conf, err = query_model(t)
            results.append(conf if conf is not None else 0.0)
        return results


def heuristic_prediction(text: str) -> float:
    disaster_terms = {
        "earthquake", "flood", "wildfire", "fire", "hurricane", 
        "evacuation", "disaster", "collapsed", "injured", "dead", 
        "tsunami", "explosion", "rescue", "storm", "collision", 
        "crash", "emergency", "alert"
    }
    words = set(re.findall(r"\w+", text.lower()))
    matches = words.intersection(disaster_terms)
    if matches:
        return min(0.9, 0.4 + (len(matches) * 0.15))
    return 0.15


def explain_prediction(text: str, base_confidence: float) -> Dict[str, float]:
    words = text.split()
    if not words:
        return {}

    # Limiter à 10 mots pour la rapidité
    words_to_test = words[:10]
    variations = []
    
    for i in range(len(words_to_test)):
        ablated = " ".join(words_to_test[:i] + words_to_test[i + 1 :])
        variations.append(ablated if ablated.strip() else "[EMPTY]")

    ablated_confidences = query_model_batch(variations)

    impacts = {}
    for i, word in enumerate(words_to_test):
        conf = ablated_confidences[i] if i < len(ablated_confidences) else base_confidence
        impacts[word] = round(base_confidence - conf, 4)

    return dict(sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True))


# --- ENDPOINTS ---

@app.on_event("startup")
def startup_event():
    """Précharge le modèle au démarrage de l'API pour éviter la latence à la première requête."""
    print("Démarrage de l'API, tentative de préchargement du modèle...")
    get_classifier()


@app.get("/")
def home():
    return {"message": "API BERT (Local Transformers) v3.0.0 active. /docs pour tester."}


@app.get("/health")
def health():
    classifier = get_classifier()
    status = "ok" if classifier is not None else "error"
    return {
        "status": status,
        "mode": "local_transformers",
        "model_loaded": classifier is not None,
        "model_name": "BERTweet (Local via Transformers)",
        "model_error": _model_load_error,
        "hf_model_id": HF_MODEL_ID,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict_tweet(tweet: TweetInput):
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

    # 4. Prédiction via pipeline local ou fallback
    confidence, error = query_model(cleaned_text)

    if error:
        confidence = heuristic_prediction(cleaned_text)
        model_used = f"Heuristic Fallback ({error[:80]})"
        impact_words: Dict[str, float] = {}
    else:
        model_used = "BERTweet (Local via Transformers)"
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
