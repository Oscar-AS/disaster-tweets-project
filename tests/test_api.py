from unittest.mock import patch

from fastapi.testclient import TestClient

from API.main import app, clean_text_advanced


client = TestClient(app)


def test_clean_text_advanced():
    assert clean_text_advanced("Fire! 🔥") == "Fire! fire"
    assert clean_text_advanced("Check this http://example.com") == "Check this"
    assert clean_text_advanced("Hello @john_doe, help us!") == "Hello [USER], help us!"
    assert (
        clean_text_advanced("Café & résumé 😊")
        == "Caf & rsum smiling face with smiling eyes"
    )


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["mode"] == "huggingface_inference"
    assert data["model_loaded"] is True


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_validation_error():
    response = client.post("/predict", json={"location": "USA"})

    assert response.status_code == 422


@patch("API.main.query_huggingface_batch")
@patch("API.main.query_huggingface")
def test_predict_success_disaster(mock_hf, mock_hf_batch):
    mock_hf.return_value = (0.92, None)
    mock_hf_batch.return_value = [0.5, 0.7, 0.8, 0.9, 0.85]

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
    assert data["clean_text"] == "Huge earthquake hits the city!"
    assert "BERTweet" in data["model_name"]
    assert data["impact_words"]


@patch("API.main.query_huggingface_batch")
@patch("API.main.query_huggingface")
def test_predict_success_not_disaster(mock_hf, mock_hf_batch):
    mock_hf.return_value = (0.12, None)
    mock_hf_batch.return_value = [0.1, 0.08, 0.09, 0.11]

    response = client.post("/predict", json={"text": "What a beautiful day!"})

    assert response.status_code == 200
    data = response.json()
    assert data["is_disaster"] is False
    assert data["confidence"] == 0.12


@patch("API.main.query_huggingface")
def test_predict_hf_error_fallback(mock_hf):
    mock_hf.return_value = (None, "Connection timeout")

    response = client.post("/predict", json={"text": "Massive earthquake and flood!"})

    assert response.status_code == 200
    data = response.json()
    assert data["is_disaster"] is True
    assert "Heuristic Fallback" in data["model_name"]


def test_predict_empty_text():
    response = client.post("/predict", json={"text": "   "})

    assert response.status_code == 200
    data = response.json()
    assert data["is_disaster"] is False
    assert data["confidence"] == 0.0
    assert data["clean_text"] == ""
