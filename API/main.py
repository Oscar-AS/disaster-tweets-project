from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import re
import emoji
import dagshub
#import os
import pandas as pd
import numpy as np
import builtins
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(name="f2_score")
def f2_score(y_true, y_pred):
    """
    Custom F2 metric used when the production Keras model was saved.

    The function must exist and be registered before MLflow/Keras loads the
    model, otherwise deserialization fails with "Could not locate function
    'f2_score'".
    """
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
tf.keras.utils.get_custom_objects()["function"] = f2_score
builtins.f2_score = f2_score

try:
    import keras

    keras.saving.register_keras_serializable(name="f2_score")(f2_score)
    keras.utils.get_custom_objects()["f2_score"] = f2_score
    keras.utils.get_custom_objects()["function"] = f2_score
except Exception:
    pass

# PATCH WINDOWS : Force l'utilisation de l'encodage UTF-8 lors de la lecture des fichiers par Keras/MLflow.
# Ceci corrige l'erreur "charmap codec can't decode byte" lors du chargement de la couche TextVectorization.
_original_open = builtins.open
def _utf8_open(*args, **kwargs):
    mode = kwargs.get('mode', args[1] if len(args) > 1 else 'r')
    if 'b' not in mode and 'encoding' not in kwargs:
        kwargs['encoding'] = 'utf-8'
    return _original_open(*args, **kwargs)
builtins.open = _utf8_open

# Initialisation de l'API
app = FastAPI(
    title="Disaster Tweet Predictor API",
    description="API de prédiction de catastrophes basée sur des tweets, propulsée par le meilleur modèle du MLflow Registry.",
    version="1.0.0"
)

# Configuration globale
MODEL_NAME = "Disaster_Tweet_Predictor_Prod"
ALIAS = "best_model"
model = None
model_run_name = "Inconnu"
model_load_error = None

@app.on_event("startup")
def load_model():
    global model, model_load_error
    
    # Initialisation de la connexion à DagsHub pour récupérer le modèle distant.
    try:
        dagshub.init(repo_owner='Oscar-AS', repo_name='disaster-tweets-project', mlflow=True)
        print("DagsHub init OK.")
    except Exception as e:
        print(f"Info DagsHub : {e}")

    # Chargement du modèle de production depuis MLflow
    try:
        print(f"Téléchargement et chargement du modèle MLflow '{MODEL_NAME}' avec l'alias '@{ALIAS}'...")
        model_uri = f"models:/{MODEL_NAME}@{ALIAS}"
        model = mlflow.pyfunc.load_model(model_uri)
        model_load_error = None
        
        # Récupération du nom du modèle (Run Name)
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(model.metadata.run_id)
        global model_run_name
        model_run_name = run.data.tags.get("mlflow.runName", "Modèle sans nom")
        
        print(f"Modèle chargé en mémoire avec succès ! Nom : {model_run_name}")
    except Exception as e:
        model = None
        model_load_error = str(e)
        print(f"Attention: Impossible de charger le modèle depuis MLflow. L'API démarrera mais /predict renverra une erreur. Détail: {e}")

# --- SCHÉMAS D'ENTRÉE ET DE SORTIE ---

class TweetInput(BaseModel):
    text: str
    location: str = None
    keyword: str = None

class PredictionOutput(BaseModel):
    is_disaster: bool
    confidence: float
    clean_text: str
    model_name: str

# --- PIPELINE DE PRÉTRAITEMENT ---

def extract_disaster_confidence(prediction) -> float:
    """
    Normalise les sorties de prediction courantes vers une probabilite
    de la classe positive (catastrophe).
    """
    if isinstance(prediction, pd.DataFrame) and 'score' in prediction.columns:
        row = prediction.iloc[0]
        if 'label' in prediction.columns and row['label'] != 'LABEL_1':
            return 1.0 - float(row['score'])
        return float(row['score'])

    if isinstance(prediction, list) and len(prediction) > 0 and isinstance(prediction[0], dict):
        res = prediction[0]
        if res.get('label') == 'LABEL_1':
            return float(res.get('score', 0))
        if 'score' in res:
            return 1.0 - float(res.get('score', 0))

    if isinstance(prediction, np.ndarray):
        return float(prediction.flatten()[0])

    return float(np.array(prediction).flatten()[0])

def clean_text_advanced(text: str) -> str:
    """
    Applique le même nettoyage que lors de la phase d'entraînement.
    """
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+", "[USER]", text)
    text = emoji.demojize(text)
    text = text.replace(":", " ")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    # Suppression des caractères non-ASCII (comme fait en entraînement pour corriger le charmap)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text

# --- ENDPOINTS ---

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API Disaster Tweets. Allez sur /docs pour tester."}

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_error": model_load_error,
    }

@app.post("/predict", response_model=PredictionOutput)
def predict_tweet(tweet: TweetInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Le modèle n'a pas pu être chargé depuis MLflow.")
        
    cleaned_text = clean_text_advanced(tweet.text)
    
    if not cleaned_text:
        return PredictionOutput(is_disaster=False, confidence=0.0, clean_text="", model_name=model_run_name)

    try:
        # Format 1 : Numpy Array (Format requis par Keras 3 et accepté par la plupart des modèles)
        input_data = np.array([cleaned_text])
        prediction = model.predict(input_data)
        confidence = extract_disaster_confidence(prediction)

        is_disaster = confidence > 0.5
        
        return PredictionOutput(
            is_disaster=is_disaster,
            confidence=confidence,
            clean_text=cleaned_text,
            model_name=model_run_name
        )
    except Exception as e:
        # En cas d'erreur de format, essai 2 : DataFrame (Format standard absolu pour Sklearn et XGBoost via MLflow)
        try:
            df_input = pd.DataFrame({"text": [cleaned_text]})
            prediction = model.predict(df_input)
            confidence = extract_disaster_confidence(prediction)
            is_disaster = confidence > 0.5
            return PredictionOutput(
                is_disaster=is_disaster,
                confidence=confidence,
                clean_text=cleaned_text,
                model_name=model_run_name
            )
        except Exception as e2:
            # Essai 3 : Extraction du modèle natif (Bypass total du wrapper strict de MLflow)
            try:
                keras_model = None
                if hasattr(model, '_model_impl'):
                    impl = model._model_impl
                    if hasattr(impl, 'keras_model'):
                        keras_model = impl.keras_model
                    elif hasattr(impl, 'get_raw_model'):
                        keras_model = impl.get_raw_model()
                
                if keras_model is None:
                    keras_model = model.unwrap_python_model()
                    
                # Keras 3 avec TensorFlow backend exige souvent un Tensor pour les chaînes de caractères
                import tensorflow as tf
                prediction = keras_model.predict(tf.constant([cleaned_text]))
                confidence = float(np.array(prediction).flatten()[0])
                is_disaster = confidence > 0.5
                return PredictionOutput(
                    is_disaster=is_disaster,
                    confidence=confidence,
                    clean_text=cleaned_text,
                    model_name=model_run_name
                )
            except Exception as e3:
                # Dernier recours ultime : la liste pure pour un transformer
                try:
                    prediction = model.predict([cleaned_text])
                    confidence = float(np.array(prediction).flatten()[0])
                    is_disaster = confidence > 0.5
                    return PredictionOutput(
                        is_disaster=is_disaster,
                        confidence=confidence,
                        clean_text=cleaned_text,
                        model_name=model_run_name
                    )
                except Exception as e4:
                    raise HTTPException(status_code=500, detail=f"Échec de tous les formats de prédiction.\n1(Numpy): {str(e)}\n2(DataFrame): {str(e2)}\n3(Keras natif): {str(e3)}\n4(Liste pure): {str(e4)}")
