"""
Tests pour l'API_BERT (Hugging Face Inference).
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from API_BERT.main import app, clean_text_advanced

client = TestClient(app)


def test_clean_text_advanced():
    """Test le nettoyage de texte."""
    assert clean_text_advanced("Fire! 🔥") == "Fire! fire"
    assert clean_text_advanced("Check this http://example.com") == "Check this"
    assert clean_text_advanced("Hello @john_doe, help us!") == "Hello [USER], help us!"


def test_health_check():
    """Test le endpoint /health."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["mode"] == "huggingface_inference"
    assert data["model_loaded"] is True


def test_home():
    """Test le endpoint /."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_validation_error():
    """Test qu'un JSON sans 'text' renvoie 422."""
    response = client.post("/predict", json={"location": "USA"})
    assert response.status_code == 422


@patch("API_BERT.main.query_huggingface")
def test_predict_success_disaster(mock_hf):
    """Test une prédiction réussie (catastrophe)."""
    mock_hf.return_value = (0.92, None)

    response = client.post(
        "/predict",
        json={
            "text": "Huge earthquake hits the city!",
            "location": "California",
            "keyword": "earthquake",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["is_disaster"] is True
    assert data["confidence"] == 0.92
    assert "model_name" in data
    assert "BERTweet" in data["model_name"]


@patch("API_BERT.main.query_huggingface")
def test_predict_success_not_disaster(mock_hf):
    """Test une prédiction réussie (pas catastrophe)."""
    mock_hf.return_value = (0.12, None)

    response = client.post("/predict", json={"text": "What a beautiful day!"})

    assert response.status_code == 200
    data = response.json()
    assert data["is_disaster"] is False
    assert data["confidence"] == 0.12


@patch("API_BERT.main.query_huggingface")
def test_predict_hf_error_fallback(mock_hf):
    """Test le fallback heuristique quand HF échoue."""
    mock_hf.return_value = (None, "Connection timeout")

    response = client.post(
        "/predict", json={"text": "Massive earthquake and flood!"}
    )

    assert response.status_code == 200
    data = response.json()
    # Le fallback heuristique devrait détecter "earthquake" et "flood"
    assert data["is_disaster"] is True
    assert "Heuristic Fallback" in data["model_name"]


def test_predict_empty_text():
    """Test avec un texte vide."""
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 200
    data = response.json()
    assert data["is_disaster"] is False
    assert data["confidence"] == 0.0
    assert data["clean_text"] == ""
