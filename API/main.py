from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import re
import emoji
import dagshub
import os
import pandas as pd
import numpy as np

# Initialisation de l'API
app = FastAPI(
    title="Disaster Tweet Predictor API",
    description="API de prédiction de désastres basée sur des tweets, propulsée par le meilleur modèle du MLflow Registry.",
    version="1.0.0"
)

# Configuration globale
MODEL_NAME = "Disaster_Tweet_Predictor_Prod"
STAGE = "Production"
model = None

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
        print(f"Téléchargement et chargement du modèle MLflow '{MODEL_NAME}' en mode '{STAGE}'...")
        model_uri = f"models:/{MODEL_NAME}/{STAGE}"
        model = mlflow.pyfunc.load_model(model_uri)
        print("Modèle chargé en mémoire avec succès !")
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
        return PredictionOutput(is_disaster=False, confidence=0.0, clean_text="")

    try:
        # Essai de format 1 : Pandas Series (format idéal pour Keras TextVectorization)
        input_data = pd.Series([cleaned_text])
        
        # Le pyfunc de mlflow s'occupe du wrapping
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
            clean_text=cleaned_text
        )
    except Exception as e:
        # En cas d'erreur de format d'entrée, tentative avec un format list classique
        try:
            prediction = model.predict([cleaned_text])
            confidence = float(np.array(prediction).flatten()[0])
            is_disaster = confidence > 0.5
            return PredictionOutput(
                is_disaster=is_disaster,
                confidence=confidence,
                clean_text=cleaned_text
            )
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"Erreur interne lors de la prédiction MLflow: {str(e)} / {str(e2)}")
