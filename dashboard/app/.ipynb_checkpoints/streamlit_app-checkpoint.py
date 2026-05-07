# Module temps pour simuler un flux en direct avec attente entre tweets.
import time
# Types pour clarifier les structures de donnees.
from typing import Any, Dict, List

# Folium dessine des cartes interactives.
import folium
# pandas facilite la preparation des donnees pour les graphiques.
import pandas as pd
# plotly.express sert a tracer les graphiques rapidement.
import plotly.express as px
# requests permet d'appeler l'API FastAPI via HTTP.
import requests 
# streamlit construit l'interface web.
import streamlit as st
# Composant visuel pour embellir les cartes de metriques.
from streamlit_extras.metric_cards import style_metric_cards
# Bridge entre Folium et Streamlit.
from streamlit_folium import st_folium

# Adresse de l'endpoint API de prediction.
API_URL = "http://localhost:8000/predict"

# Exemples de tweets utilises pour la simulation en continu.
SIMULATED_TWEETS = [
    "Massive wildfire spreading quickly near Madrid suburbs, evacuation underway!",
    "Beautiful sunset in Paris, everyone is enjoying the evening.",
    "Flood warnings issued in Lagos after heavy overnight rain.",
    "Concert tonight in Berlin was amazing and packed.",
    "Explosion reported downtown New York, emergency teams dispatched.",
]


# Configure l'apparence generale de la page.
def setup_page() -> None:
    # Titre navigateur + mise en page large.
    st.set_page_config(page_title="Disaster Tweets Dashboard", layout="wide")
    # Injection CSS simple pour forcer un theme sombre.
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Titre principal visible dans l'application.
    st.title("Detection de Tweets de Catastrophe")
    # Sous-titre court.
    st.caption("Dashboard MLOps connecte a FastAPI")


# Envoie un tweet a l'API et recupere la prediction en JSON.
def call_api(tweet_text: str) -> Dict[str, Any]:
    # Appel HTTP POST vers l'API avec timeout de securite.
    response = requests.post(API_URL, json={"text": tweet_text}, timeout=30)
    # Leve une erreur si le statut HTTP n'est pas 2xx.
    response.raise_for_status()
    # Retourne la reponse convertie en dictionnaire Python.
    return response.json()


# Affiche un graphique des mots qui influencent le plus la prediction.
def plot_impact_words(impact_words: Dict[str, float]) -> None:
    # Si pas de donnees, on informe l'utilisateur.
    if not impact_words:
        st.info("Aucune contribution de mot disponible.")
        return
    # Conversion du dictionnaire en tableau pour Plotly.
    df = pd.DataFrame(
        {"word": list(impact_words.keys()), "impact": list(impact_words.values())}
    ).sort_values("impact", ascending=False)
    # Creation du graphique en barres.
    fig = px.bar(
        df,
        x="word",
        y="impact",
        title="Contribution des mots a la prediction",
        color="impact",
        color_continuous_scale="RdBu",
    )
    # Theme sombre du graphique.
    fig.update_layout(template="plotly_dark")
    # Affichage dans Streamlit.
    st.plotly_chart(fig, use_container_width=True)


# Affiche la carte des lieux detectes.
def plot_map(geo_coords: List[List[float]]) -> None:
    # Centre de la carte: premier point detecte ou vue globale par defaut.
    center = geo_coords[0] if geo_coords else [20.0, 0.0]
    # Carte sombre avec zoom global.
    fmap = folium.Map(location=center, zoom_start=2, tiles="CartoDB dark_matter")
    # Ajoute un marqueur pour chaque coordonnee detectee.
    for lat, lon in geo_coords:
        folium.Marker([lat, lon]).add_to(fmap)
    # Rend la carte dans l'interface Streamlit.
    st_folium(fmap, width=900, height=400)


# Lance une simulation "live" en lisant des tweets predefinis un par un.
def simulate_stream() -> None:
    # Titre de section.
    st.subheader("Simulation de flux en direct")
    # Zone dynamique qui sera remplacee a chaque nouvelle prediction.
    placeholder = st.empty()

    # Le traitement commence quand l'utilisateur clique le bouton.
    if st.button("Lancer la simulation"):
        # On parcourt chaque tweet de test.
        for tweet in SIMULATED_TWEETS:
            # Le contenu est redessine dans la meme zone (effet flux).
            with placeholder.container():
                # Affiche le tweet courant.
                st.markdown(f"**Tweet:** {tweet}")
                try:
                    # Appel API pour obtenir la prediction.
                    pred = call_api(tweet)
                    # Deux colonnes pour les metriques principales.
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Prediction catastrophe", str(pred["is_disaster"]))
                    with col2:
                        st.metric("Score", f'{pred["score"]:.4f}')
                    # Habillage visuel des cartes de metriques.
                    style_metric_cards()
                    # Affichage de l'explicabilite.
                    plot_impact_words(pred.get("impact_words", {}))
                    # Affichage de la carte geographique.
                    plot_map(pred.get("geo_coords", []))
                except Exception as exc:
                    # Message clair si l'API n'est pas joignable ou en erreur.
                    st.error(f"Erreur API: {exc}")
            # Pause pour simuler un flux temps reel.
            time.sleep(2)


# Zone de prediction manuelle: l'utilisateur colle son propre tweet.
def manual_prediction() -> None:
    # Titre de section.
    st.subheader("Prediction manuelle")
    # Champ texte multiligne.
    user_text = st.text_area("Entrez un tweet (5-280 caracteres)", height=120)
    # Le calcul part au clic sur le bouton.
    if st.button("Predire"):
        try:
            # Appel API avec le texte saisi.
            pred = call_api(user_text)
            # Affiche deux metriques cles.
            c1, c2 = st.columns(2)
            c1.metric("is_disaster", str(pred["is_disaster"]))
            c2.metric("score", f'{pred["score"]:.4f}')
            style_metric_cards()

            # Zone gauche = explication, zone droite = carte.
            left, right = st.columns([2, 3])
            with left:
                plot_impact_words(pred.get("impact_words", {}))
            with right:
                plot_map(pred.get("geo_coords", []))
        except Exception as exc:
            # Message d'erreur utilisateur.
            st.error(f"Echec prediction: {exc}")


# Fonction principale: assemble toutes les sections de la page.
def main() -> None:
    setup_page()
    manual_prediction()
    # Ligne de separation visuelle.
    st.divider()
    simulate_stream()


# Ce bloc execute main() seulement si le fichier est lance directement.
if __name__ == "__main__":
    main()

