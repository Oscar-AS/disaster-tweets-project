# Disaster Tweet Detection - MLOps Modular System

Systeme complet de detection de tweets de catastrophe avec:
- API FastAPI de prediction
- Dashboard Streamlit connecte a l'API
- Coeur NLP modulaire (spaCy + modele `.joblib`)
- Dockerisation des services

## Architecture

```text
ML/
├── api/
│   └── main.py
├── app/
│   └── streamlit_app.py
├── docker/
│   └── start_services.sh
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── processor.py
│   └── schemas.py
├── Dockerfile
├── README.md
└── requirements.txt
```

## Fonctionnalites

### API (`api/main.py`)
- Endpoint `POST /predict`
- Validation stricte via Pydantic:
  - texte obligatoire
  - longueur entre 5 et 280 caracteres
  - rejet des textes vides apres nettoyage
- Reponse:
  - `is_disaster` (bool)
  - `score` (float)
  - `impact_words` (dict interpretable type SHAP approx.)
  - `geo_coords` (liste de coordonnees extraites via NER)

### Dashboard (`app/streamlit_app.py`)
- Theme sombre
- Layout large (`wide`)
- Cartes de metriques stylees (`streamlit-extras`)
- Simulation de flux live via `st.empty()` + delai de traitement
- Carte geospatiale (`streamlit-folium`) avec tuiles DarkMatter
- Graphique Plotly en barres pour la contribution des mots
- Chaque prediction passe par l'API FastAPI (appel HTTP)

### Coeur ML (`src/`)
- `processor.py`: nettoyage texte + extraction de lieux via spaCy NER
- `model.py`: inference via modele `.joblib` + score + impact mots

## Utiliser ton modele `.joblib`

Par defaut, l'application charge le modele depuis:
- `models/disaster_model.joblib`

Tu peux aussi definir un chemin personnalise avec la variable d'environnement:

```bash
# Linux/Mac
export MODEL_PATH="models/mon_modele.joblib"

# Windows PowerShell
$env:MODEL_PATH="models/mon_modele.joblib"
```

Interface attendue pour le modele:
- idealement un pipeline scikit-learn avec `predict_proba`
- sinon `decision_function` ou `predict`

## Installation locale

1. Creer un environnement virtuel:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

2. Installer les dependances:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Lancement local

### 1) API
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2) Dashboard (dans un 2e terminal)
```bash
streamlit run app/streamlit_app.py --server.port 8501
```

Acces:
- API docs: `http://localhost:8000/docs`
- Streamlit: `http://localhost:8501`

## Lancement avec Docker

Construire l'image:
```bash
docker build -t disaster-mlops .
```

Executer le container:
```bash
docker run --rm -p 8000:8000 -p 8501:8501 disaster-mlops
```

## Exemple payload API

```json
{
  "text": "Huge flood in Lagos, rescue teams are trying to evacuate families."
}
```

## Notes MLOps

- Separation claire des couches (API / UI / logique ML)
- Interfaces stables via schemas Pydantic
- Pret pour extension:
  - remplacement par un modele fine-tune NLP plus avance si besoin
  - ajout de tracking MLflow
  - ajout de tests unitaires/CI

