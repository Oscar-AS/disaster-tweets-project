# 🌪️ Disaster Tweets — Détection de Tweets de Catastrophe

[![CI](https://github.com/Oscar-AS/disaster-tweets-project/actions/workflows/ci.yml/badge.svg)](https://github.com/Oscar-AS/disaster-tweets-project/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Contexte

Twitter est devenu un canal de communication critique lors des catastrophes naturelles (séismes, inondations, incendies…). Des millions de tweets sont publiés chaque minute, mais tous ne signalent pas une vraie urgence. Ce projet répond à une mission confiée par **Twitter** : construire un modèle de Machine Learning capable de **distinguer automatiquement les tweets relatifs à de vraies catastrophes** de ceux qui ne le sont pas.

La mission complète comprend :
1. **Modélisation ML** — entraîner et comparer des modèles de classification
2. **MLOps** — suivi des expériences, traçabilité, CI/CD
3. **API de prédiction** — déploiement via FastAPI sur Hugging Face Spaces
4. **Dashboard** — interface de démonstration Streamlit

---

## 🗂️ Structure du dépôt

```
disaster-tweets-project/
│
├── 📁 Notebooks/                      # Travail ML principal (EDA → sélection)
│   ├── disastertweets_01_eda_baselines.ipynb      # EDA + modèles de base
│   ├── disastertweets_02_model_BiLSTM.ipynb       # Modèle BiLSTM
│   ├── disastertweets_02_model_LSTM.ipynb         # Modèle LSTM
│   ├── disastertweets_02_model_TextCNN.ipynb      # Modèle TextCNN
│   ├── disastertweets_03_model_BERT.ipynb         # Fine-tuning BERT
│   ├── disastertweets_03_model_DistilBERT.ipynb   # Fine-tuning DistilBERT
│   └── disastertweets_04_selection_finale.ipynb   # Sélection du modèle final
│
├── 📁 API/                            # API FastAPI de prédiction
│   ├── main.py                        # Application FastAPI (v3 — BERTweet local)
│   ├── requirements.txt               # Dépendances légères de l'API
│   └── .env                           # Variables d'environnement (non versionné)
│
├── 📁 dashboard/                      # Interface Streamlit
│   ├── app/streamlit_app.py           # Application principale
│   ├── src/                           # Modules auxiliaires
│   ├── Dockerfile                     # Image Docker du dashboard
│   └── requirements.txt
│
├── 📁 hf_space/                       # Configuration Hugging Face Space (API)
│   ├── Dockerfile                     # Déploiement de l'API sur HF Spaces
│   └── README.md
│
├── 📁 scripts/
│   └── push_to_hf.py                  # Script de push vers Hugging Face Hub
│
├── 📁 tests/
│   └── test_api.py                    # Tests unitaires de l'API (pytest)
│
├── 📁 Documents/                      # Captures MLflow et ressources de doc
├── 📁 Base/
│   └── tweets.csv                     # Données brutes Kaggle
│
├── 📁 .github/workflows/
│   ├── ci.yml                         # Pipeline CI (ruff + mypy + pytest)
│   └── sync_to_hf_space.yml          # Sync automatique vers Hugging Face
│
├── requirements.txt                   # Dépendances complètes (notebooks + API)
├── .python-version                    # Version Python cible
├── .gitignore
└── LICENSE
```

---

## 🔬 Approche ML — Les Notebooks

La progression suit une logique claire, du plus simple au plus complexe.

### Notebook 01 — EDA & Modèles de base
**`disastertweets_01_eda_baselines.ipynb`**

- Analyse exploratoire des données (distribution des classes, fréquences de mots, nuages de mots)
- Prétraitement du texte : suppression des URLs, mentions, emojis, normalisation
- Entraînement des **modèles de base** : Logistic Regression, SVM, Random Forest, Naive Bayes
- Premiers modèles ensemblistes (Voting, Stacking)
- Évaluation initiale sur la précision et le **F2-score**

### Notebooks 02 — Deep Learning séquentiel
**`disastertweets_02_model_*.ipynb`**

Trois architectures de réseaux de neurones entraînées sur des embeddings TF-IDF / Word2Vec :
- **LSTM** (Long Short-Term Memory)
- **BiLSTM** (Bidirectional LSTM)
- **TextCNN** (Convolutional Neural Network pour le texte)

### Notebooks 03 — Transformers pré-entraînés
**`disastertweets_03_model_*.ipynb`**

Fine-tuning de modèles Transformers Hugging Face :
- **BERT** (`bert-base-uncased`)
- **DistilBERT** (`distilbert-base-uncased`) avec interprétabilité SHAP

### Notebook 04 — Sélection finale
**`disastertweets_04_selection_finale.ipynb`**

Comparaison finale de tous les modèles via **MLflow (DagsHub)** sur les métriques consolidées.

**Critère de sélection principal : le F2-Score**
> Le F2-score est privilégié sur le F1-score car dans le contexte des catastrophes, **il faut absolument minimiser les faux négatifs** (un vrai tweet d'urgence non détecté peut coûter des vies). Le F2-score pénalise davantage le rappel raté que la précision.

**Modèle final choisi : `BERTweet`** (`vinai/bertweet-base`)
- Meilleur F2-score et meilleur rappel sur la classe "catastrophe"
- Pré-entraîné spécifiquement sur des tweets — parfaitement adapté au domaine
- Interprétabilité via SHAP disponible

---

## ⚙️ MLOps

### Suivi des expériences — MLflow + DagsHub
Toutes les expériences sont tracées avec **MLflow** hébergé sur **DagsHub** :
- Métriques : Accuracy, F1, F2-score, Précision, Rappel, AUC
- Paramètres des modèles (learning rate, epochs, etc.)
- Artefacts : modèles sauvegardés, courbes Precision-Recall, courbes de Lift

### CI/CD — GitHub Actions
Deux workflows automatisés :

| Workflow | Déclencheur | Actions |
|---|---|---|
| `ci.yml` | push / pull request | Lint (ruff), Type check (mypy), Tests (pytest) |
| `sync_to_hf_space.yml` | push sur `main` | Synchronisation du code vers Hugging Face Spaces |

---

## 🚀 API de prédiction

**FastAPI** — déployée sur **Hugging Face Spaces**

### Fonctionnement
L'API charge le modèle BERTweet **localement** via la bibliothèque `transformers` (pas d'appel externe à l'inférence Hugging Face). Elle intègre :

1. **Détection de langue + traduction automatique** (`langdetect` + `deep-translator`) — supporte les tweets non-anglophones
2. **Nettoyage du texte** — suppression des URLs, normalisation des mentions (`@user → [USER]`), conversion des emojis
3. **Prédiction BERTweet** — classification binaire (catastrophe / pas catastrophe)
4. **Explicabilité par ablation** — importance de chaque mot via suppression et mesure d'impact sur le score
5. **Fallback heuristique** — si le modèle ne charge pas, un système basé sur un lexique de mots-clés prend le relai

### Endpoints

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/` | Statut de l'API |
| `GET` | `/health` | Santé + état du modèle |
| `POST` | `/predict` | Prédiction sur un tweet |

**Exemple de requête :**
```bash
curl -X POST "https://<api-url>/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Huge earthquake hits the city!", "location": "California"}'
```

**Exemple de réponse :**
```json
{
  "is_disaster": true,
  "confidence": 0.92,
  "clean_text": "Huge earthquake hits the city!",
  "model_name": "BERTweet (Local via Transformers)",
  "impact_words": {"earthquake": 0.34, "hits": 0.12, "city": 0.05},
  "detected_lang": "en",
  "translated_text": "Huge earthquake hits the city!"
}
```

---

## 📊 Dashboard Streamlit

Interface web de démonstration déployée via Docker.

**Fonctionnalités :**
- 🔐 Connexion sécurisée par mot de passe
- 📝 **Analyse manuelle** d'un tweet avec score de confiance et importance des mots
- 📂 **Analyse par lot (CSV)** avec téléchargement des résultats
- 🌍 Traduction automatique des tweets non-anglophones
- 📡 Indicateur d'état de l'API en temps réel

---

## 🛠️ Installation locale

### Prérequis
- Python 3.11+
- `git`

### 1. Cloner le dépôt
```bash
git clone https://github.com/Oscar-AS/disaster-tweets-project.git
cd disaster-tweets-project
```

### 2. Créer un environnement virtuel
```bash
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# Linux / Mac
source .venv/bin/activate
```

### 3. Installer les dépendances

Pour les notebooks (ML complet) :
```bash
pip install -r requirements.txt
```

Pour l'API seule (léger) :
```bash
pip install -r API/requirements.txt
```

### 4. Lancer l'API en local
```bash
uvicorn API.main:app --reload
```
→ Disponible sur `http://localhost:8000/docs`

### 5. Lancer le dashboard en local
```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app/streamlit_app.py
```
→ Disponible sur `http://localhost:8501`

---

## 🧪 Tests

```bash
pytest tests/
```

Les tests couvrent :
- Nettoyage du texte (`clean_text_advanced`)
- Endpoints `/`, `/health`, `/predict`
- Cas limites : texte vide, erreur modèle (fallback heuristique), erreur de validation

---

## 📦 Données

Les données proviennent du challenge **[Kaggle — Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)**.

- `Base/tweets.csv` — Dataset brut (train + test)
- Classes : `1` = tweet de catastrophe réelle, `0` = tweet non lié à une catastrophe

---

## 📈 Résultats clés

| Modèle | F2-Score | Rappel (classe 1) |
|---|---|---|
| Logistic Regression (baseline) | ~0.72 | ~0.74 |
| LSTM / BiLSTM / TextCNN | ~0.75–0.78 | ~0.77–0.80 |
| BERT / DistilBERT | ~0.80–0.83 | ~0.82–0.85 |
| **BERTweet (modèle final)** | **~0.85+** | **~0.87+** |

> *Métriques issues du suivi MLflow sur DagsHub. Voir `Notebooks/disastertweets_04_selection_finale.ipynb` pour la comparaison complète.*

---

## 🤝 Équipe

Projet réalisé dans le cadre du cours de **Machine Learning** — ISE2 S2, Informatique.

### Collaborateurs

| Nom | GitHub |
|-----|--------|
| DJERAKEI MISTALENGAR | [@Yves](https://github.com/DJERAKEI221) |
| Mame Balla Bousso | [@Bousso](https://github.com/MameBallaBousso) |
| Samba SOW | [@Samba](https://github.com/Samba99Sow) |
| KAFANDO Oscar | [@Oscar](https://github.com/Oscar-AS) |

---

## 📄 Licence

Ce projet est sous licence [MIT](LICENSE).
