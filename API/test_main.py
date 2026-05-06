from fastapi.testclient import TestClient
from main import app, clean_text_advanced
import pytest
from unittest.mock import patch
import pandas as pd
import numpy as np

client = TestClient(app)

def test_clean_text_advanced():
    # Test la traduction d'émoji et nettoyage de base
    assert clean_text_advanced("Fire! 🔥") == "Fire!  fire"
    
    # Test la suppression des URLs
    assert clean_text_advanced("Check this http://example.com") == "Check this"
    
    # Test la conversion des mentions
    assert clean_text_advanced("Hello @john_doe, help us!") == "Hello [USER], help us!"
    
    # Test la suppression des caractères spéciaux non-ASCII (et de la ponctuation conservée si ascii)
    assert clean_text_advanced("Café & résumé 😊") == "Caf & rsum  smiling face with smiling eyes"
    # L'émoji est d'abord traduit en ASCII par demojize, puis les accents (non-ascii) sautent.
    # On vérifie juste que ça ne plante pas et que c'est bien nettoyé.

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "ok"

def test_predict_validation_error():
    # Si on envoie un JSON vide (sans le champ requis 'text')
    response = client.post("/predict", json={"location": "USA"})
    assert response.status_code == 422 # Unprocessable Entity (Pydantic validation error)

@patch('main.model')
def test_predict_endpoint_success(mock_model):
    # On mocke le comportement de notre modèle MLflow pour éviter un vrai appel (et pour tester l'API de manière isolée)
    # Imaginons que ce soit un modèle Keras qui renvoie un NumPy array de probabilités
    mock_model.predict.return_value = np.array([[0.85]])
    
    response = client.post("/predict", json={
        "text": "Huge earthquake hits the city!",
        "location": "California",
        "keyword": "earthquake"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["is_disaster"] is True
    assert data["confidence"] == 0.85
    assert data["clean_text"] == "Huge earthquake hits the city!"

@patch('main.model')
def test_predict_endpoint_success_transformers_format(mock_model):
    # Imaginons que ce soit un modèle Hugging Face qui renvoie un DataFrame
    mock_model.predict.return_value = pd.DataFrame([{"label": "LABEL_1", "score": 0.95}])
    
    response = client.post("/predict", json={
        "text": "Building on fire"
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["is_disaster"] is True
    assert data["confidence"] == 0.95
    assert data["clean_text"] == "Building on fire"

@patch('main.model')
def test_predict_endpoint_empty_text(mock_model):
    # Si le texte est composé uniquement de vide ou d'emojis ignorés et devient vide après nettoyage
    # Le comportement défini est de retourner directement False sans appeler le modèle
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 200
    data = response.json()
    assert data["is_disaster"] is False
    assert data["confidence"] == 0.0
    assert data["clean_text"] == ""
    # On vérifie que le modèle n'a pas été appelé
    mock_model.predict.assert_not_called()
