from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import re
import emoji
import dagshub
import os
import pandas as pd
import numpy as np
import builtins

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

@app.on_event("startup")
def load_model():
    global model
    
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
        
        # Récupération du nom du modèle (Run Name)
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(model.metadata.run_id)
        global model_run_name
        model_run_name = run.data.tags.get("mlflow.runName", "Modèle sans nom")
        
        print(f"Modèle chargé en mémoire avec succès ! Nom : {model_run_name}")
    except Exception as e:
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

def clean_text_advanced(text: str) -> str:
    """
    Applique le même nettoyage que lors de la phase d'entraînement.
    """
    text = emoji.demojize(text)
    text = text.replace(":", " ")
    text = text.replace("_", " ")
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+", "[USER]", text)
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
    return {"status": "ok", "model_loaded": model is not None}

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
        
        confidence = 0.0
        
        # Logique défensive pour interpréter la sortie peu importe l'architecture (Keras ou Transformers)
        
        # Cas 1 : Hugging Face Transformers
        # La sortie d'un pipeline Transformers encapsulé par MLflow ressemble souvent à un DataFrame avec des colonnes 'label' et 'score'
        # ou à une liste de dictionnaires.
        if isinstance(prediction, pd.DataFrame) and 'score' in prediction.columns:
            row = prediction.iloc[0]
            if row['label'] == 'LABEL_1':
                confidence = float(row['score'])
            else:
                confidence = 1.0 - float(row['score'])
                
        elif isinstance(prediction, list) and len(prediction) > 0 and isinstance(prediction[0], dict):
            res = prediction[0]
            if res.get('label') == 'LABEL_1':
                confidence = float(res.get('score', 0))
            else:
                confidence = 1.0 - float(res.get('score', 0))
                
        # Cas 2 : Keras (Sequential avec Dense final en Sigmoid)
        # La sortie est généralement un NumPy Array 2D contenant des probabilités
        elif isinstance(prediction, np.ndarray):
            confidence = float(prediction[0][0])
            
        else:
            # Fallback générique
            confidence = float(np.array(prediction).flatten()[0])

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
            confidence = float(np.array(prediction).flatten()[0])
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
