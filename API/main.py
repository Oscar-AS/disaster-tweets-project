import builtins
import gc
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional

import emoji
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Reduce TF noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["AUTOGRAPH_VERBOSITY"] = "0"

# --- CUSTOM METRICS & PATCHES ---


@lru_cache(maxsize=1)
def register_custom_metrics():
    """Lazy register custom metrics to save memory if TF isn't needed."""
    try:
        import tensorflow as tf
        
        @tf.keras.utils.register_keras_serializable(name="f2_score")
        def f2_score(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred > 0.5, tf.float32)
            true_positives = tf.reduce_sum(y_true * y_pred)
            false_positives = tf.reduce_sum((1.0 - y_true) * y_pred)
            false_negatives = tf.reduce_sum(y_true * (1.0 - y_pred))
            beta_squared = 4.0
            numerator = (1.0 + beta_squared) * true_positives
            denominator = numerator + beta_squared * false_negatives + false_positives
            return tf.math.divide_no_nan(numerator, denominator)

        tf.keras.utils.get_custom_objects()["f2_score"] = f2_score
        builtins.f2_score = f2_score
        
        import keras
        keras.saving.register_keras_serializable(name="f2_score")(f2_score)
        keras.utils.get_custom_objects()["f2_score"] = f2_score
    except Exception:
        pass

# PATCH WINDOWS: Force UTF-8 encoding
_original_open = builtins.open


def _utf8_open(*args, **kwargs):
    mode = kwargs.get("mode", args[1] if len(args) > 1 else "r")
    if "b" not in mode and "encoding" not in kwargs:
        kwargs["encoding"] = "utf-8"
    return _original_open(*args, **kwargs)


builtins.open = _utf8_open

# --- TRANSLATION UTILS ---

try:
    from deep_translator import GoogleTranslator
    from langdetect import detect
except ImportError:
    # Fallback if not installed (should be added to requirements)
    detect = None
    GoogleTranslator = None


@lru_cache(maxsize=128)
def translate_text(text: str) -> Dict[str, str]:
    """
    Detects language and translates to English if necessary.
    Uses LRU cache to speed up repeated identical requests.
    """
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


# --- API INITIALIZATION ---

app = FastAPI(
    title="Disaster Tweet Predictor API",
    description="API de prédiction de catastrophes optimisée avec BERT et explicabilité.",
    version="1.1.0",
)

MODEL_NAME = "Disaster_Tweet_Predictor_Prod"
ALIAS = "best_model"
model = None
model_run_name = "Inconnu"
model_load_error = None


def load_mlflow_model():
    """Logic to load the model from MLflow registry with memory optimization."""
    global model, model_run_name, model_load_error
    try:
        import dagshub
        import mlflow.pyfunc
        
        # Free memory before loading
        gc.collect()
        
        register_custom_metrics()
        
        dagshub.init(
            repo_owner="Oscar-AS", repo_name="disaster-tweets-project", mlflow=True
        )
        model_uri = f"models:/{MODEL_NAME}@{ALIAS}"
        
        # Load model
        model = mlflow.pyfunc.load_model(model_uri)

        # Get Run Name
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(model.metadata.run_id)
        model_run_name = run.data.tags.get("mlflow.runName", "Modèle sans nom")
        model_load_error = None
        
        # Free memory after loading
        gc.collect()
        print(f"Modèle chargé : {model_run_name}")
    except Exception as e:
        model_load_error = str(e)
        print(f"Erreur chargement modèle : {e}")
        gc.collect()


@app.on_event("startup")
def startup_event():
    load_mlflow_model()


@app.post("/refresh")
def refresh_model():
    """Endpoint to manually trigger model reload (e.g. after MLflow alias update)."""
    load_mlflow_model()
    if model_load_error:
        raise HTTPException(
            status_code=500, detail=f"Échec du rechargement : {model_load_error}"
        )
    return {"status": "success", "model_name": model_run_name}


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


def clean_text_advanced(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+", "[USER]", text)
    text = emoji.demojize(text)
    text = text.replace(":", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.encode("ascii", "ignore").decode("ascii")


def get_confidence_single(prediction_item) -> float:
    """Extracts confidence from a single prediction item (could be dict, list, scalar)."""
    try:
        if isinstance(prediction_item, dict):
            score = float(prediction_item.get("score", 0))
            label = str(prediction_item.get("label", "")).upper()
            if "LABEL_0" in label or "NON" in label or "0" == label:
                return 1.0 - score
            return score

        arr = np.array(prediction_item).flatten()
        if arr.size > 0:
            # If it's a probability [p0, p1], take p1
            if arr.size >= 2 and np.all(arr >= 0) and np.all(arr <= 1):
                return float(arr[1])
            return float(arr[0])
        return 0.0
    except Exception:
        return 0.0


def get_confidence(prediction) -> float:
    """Extracts confidence from MLflow return formats (DataFrame, list, array)."""
    if isinstance(prediction, pd.DataFrame):
        # Case: DataFrame from transformers or sklearn
        if "score" in prediction.columns:
            row = prediction.iloc[0]
            return get_confidence_single(row.to_dict())
        # Case: DataFrame with raw probabilities
        return get_confidence_single(prediction.iloc[0].values)

    if isinstance(prediction, list):
        if not prediction:
            return 0.0
        return get_confidence_single(prediction[0])

    return get_confidence_single(prediction)


def predict_batch(texts: List[str]) -> List[float]:
    """Predicts multiple texts at once for speed (used for importance)."""
    if not texts or model is None:
        return [0.0] * len(texts)
    try:
        # Prioritize batching (fast for BERT/LSTM on GPU/CPU)
        prediction = model.predict(np.array(texts))
        if isinstance(prediction, pd.DataFrame):
            return [
                get_confidence_single(row.to_dict())
                if "score" in prediction.columns
                else get_confidence_single(row.values)
                for _, row in prediction.iterrows()
            ]
        if isinstance(prediction, (list, np.ndarray)):
            return [get_confidence_single(p) for p in prediction]
        return [0.0] * len(texts)
    except Exception:
        # Fallback to sequential if the model doesn't like batching
        return [predict_internal(t) for t in texts]


def predict_internal(text: str) -> float:
    """Fast internal prediction for a single text."""
    if not text or model is None:
        return 0.0
    try:
        # Try different formats to accommodate BERT, LSTM, Sklearn, etc.
        # Format 1: Numpy array (best for BERT/Keras)
        prediction = model.predict(np.array([text]))
        return get_confidence(prediction)
    except Exception:
        try:
            # Format 2: DataFrame (best for Sklearn/XGBoost)
            prediction = model.predict(pd.DataFrame({"text": [text]}))
            return get_confidence(prediction)
        except Exception:
            try:
                # Format 3: List of strings
                prediction = model.predict([text])
                return get_confidence(prediction)
            except Exception:
                return 0.0


def explain_prediction(text: str, base_confidence: float) -> Dict[str, float]:
    """Calculates word importance using occlusion (ablation) with batch optimization."""
    words = text.split()
    if not words:
        return {}

    # Limit to 12 words for optimal speed/accuracy trade-off
    words_to_test = words[:12]

    # Generate variations (each variation is the text with one word removed)
    variations = []
    for i in range(len(words_to_test)):
        ablated = " ".join(words_to_test[:i] + words_to_test[i + 1 :])
        variations.append(ablated if ablated.strip() else "[EMPTY]")

    # Predict all variations in a single batch call (MUCH FASTER than 12 separate calls)
    ablated_confidences = predict_batch(variations)

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
    return {"message": "API Disaster Tweets v1.1.0 active. /docs pour tester."}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": model_run_name,
        "model_error": model_load_error,
    }


def heuristic_prediction(text: str) -> float:
    """Lightweight rule-based fallback if ML model is too heavy for RAM."""
    disaster_terms = {
        "earthquake", "flood", "wildfire", "fire", "hurricane", "evacuation",
        "disaster", "collapsed", "injured", "dead", "tsunami", "explosion",
        "rescue", "storm", "collision", "crash", "emergency", "alert"
    }
    words = set(re.findall(r"\w+", text.lower()))
    matches = words.intersection(disaster_terms)
    if matches:
        # Confidence increases with number of matches
        return min(0.9, 0.4 + (len(matches) * 0.15))
    return 0.15


@app.post("/predict", response_model=PredictionOutput)
def predict_tweet(tweet: TweetInput):
    # 1. Translation
    trans_res = translate_text(tweet.text)
    work_text = trans_res["translated_text"]

    # 2. Cleaning
    cleaned_text = clean_text_advanced(work_text)
    
    # 3. Prediction (ML or Heuristic Fallback)
    if model is not None:
        try:
            confidence = predict_internal(cleaned_text)
            model_used = model_run_name
            # 4. Explanation (Word Importance) - Only if ML model
            impact_words = explain_prediction(cleaned_text, confidence)
        except Exception:
            confidence = heuristic_prediction(cleaned_text)
            model_used = "Heuristic Fallback (ML Error)"
            impact_words = {}
    else:
        confidence = heuristic_prediction(cleaned_text)
        model_used = f"Heuristic Fallback (Model not loaded: {model_load_error or 'Unknown'})"
        impact_words = {}

    if not cleaned_text:
        return PredictionOutput(
            is_disaster=False,
            confidence=0.0,
            clean_text="",
            model_name=model_used,
            impact_words={},
            detected_lang=trans_res["detected_lang"],
            translated_text=work_text,
        )

    return PredictionOutput(
        is_disaster=confidence >= 0.5,
        confidence=confidence,
        clean_text=cleaned_text,
        model_name=model_used,
        impact_words=impact_words,
        detected_lang=trans_res["detected_lang"],
        translated_text=work_text,
    )
