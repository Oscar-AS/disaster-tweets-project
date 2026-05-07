# Indique que ce script doit etre execute avec bash.
#!/usr/bin/env bash
# "set -e" arrete le script immediatement si une commande echoue.
set -e

# Lance l'API FastAPI en arriere-plan sur le port 8000.
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
# Lance Streamlit au premier plan sur le port 8501 (le conteneur reste vivant grace a ce processus).
streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0

