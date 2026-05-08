# =============================================================================
# Image Docker — pour un lecteur non technique
# =============================================================================
# Une « image » est une boîte logicielle prête à l'emploi : système + Python +
# vos dépendances. Ce fichier décrit comment construire cette boîte pour faire
# tourner votre projet (API, Streamlit, etc.) de la même manière partout.
# =============================================================================

# Image de base : Linux léger avec Python 3.11 déjà installé
FROM python:3.11-slim

# Dossier de travail à l'intérieur du conteneur (comme un bureau pour les fichiers)
WORKDIR /workspace

# Copie d'abord la liste des bibliothèques, puis installation (optimise le cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Modèle linguistique anglais pour spaCy (analyse de texte)
RUN python -m spacy download en_core_web_sm

# Copie tout le code du projet dans l'image
COPY . .
# Autorise l'exécution du script de démarrage (double-clic équivalent en ligne de commande)
RUN chmod +x docker/start_services.sh

# Ports utilisés : 8000 souvent pour une API, 8501 pour Streamlit
EXPOSE 8000 8501

# Commande lancée automatiquement au démarrage du conteneur
CMD ["./docker/start_services.sh"]
