import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import folium
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_folium import st_folium

# Dossier racine du projet
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.bootstrap import ensure_packages  # noqa: E402

# Si des paquets manquent sur la machine, ils sont installés automatiquement (pip)
ensure_packages(
    [
        ("streamlit", "streamlit"),
        ("streamlit-extras", "streamlit_extras"),
        ("streamlit-folium", "streamlit_folium"),
        ("folium", "folium"),
        ("pandas", "pandas"),
        ("plotly", "plotly"),
        ("requests", "requests"),
    ]
)

# Adresse par défaut du service de prédiction
API_URL = "https://disaster-tweets-project.onrender.com/predict"
REFRESH_URL = "https://disaster-tweets-project.onrender.com/refresh"

# Trois mots de passe acceptés
ADMIN_PASSWORDS = {
    os.getenv("ADMIN_PASSWORD_1", "ADMIN1"),
    os.getenv("ADMIN_PASSWORD_2", "ADMIN2"),
    os.getenv("ADMIN_PASSWORD_3", "ADMIN3"),
}

# Villes connues pour le placement sur la carte
KNOWN_LOCATION_COORDS = {
    "paris": [48.8566, 2.3522],
    "london": [51.5072, -0.1276],
    "new york": [40.7128, -74.0060],
    "tokyo": [35.6762, 139.6503],
    "madrid": [40.4168, -3.7038],
    "lagos": [6.5244, 3.3792],
    "dakar": [14.7167, -17.4677],
    "nairobi": [-1.2864, 36.8172],
    "delhi": [28.6139, 77.2090],
    "berlin": [52.5200, 13.4050],
}


def setup_page() -> None:
    st.set_page_config(page_title="Disaster Intelligence", layout="wide")
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(circle at 15% 20%, #dbeafe 0%, transparent 30%),
                            radial-gradient(circle at 80% 0%, #cffafe 0%, transparent 35%),
                            #f8fafc;
            }
            .block-container { padding-top: 1.4rem; }
            .hero { background: linear-gradient(115deg,#312e81 0%,#4f46e5 45%,#0891b2 100%); color:#f8fafc; border-radius:16px; padding:1.1rem 1.2rem; margin-bottom:.9rem; text-align:center; }
            .card { background:#ffffff; border:1px solid #e2e8f0; border-radius:14px; padding:.9rem 1rem; box-shadow:0 4px 16px rgba(15,23,42,.06); }
            .title-row { display:flex; align-items:center; gap:.55rem; margin:.2rem 0 .7rem 0; color:#0f172a; font-weight:700; font-size:1.05rem; }
            .icon { width:26px; height:26px; border-radius:8px; display:inline-flex; align-items:center; justify-content:center; color:#fff; font-size:13px; font-weight:700; }
            .icon-alert { background:linear-gradient(135deg,#2563eb,#06b6d4); } .icon-live { background:linear-gradient(135deg,#1d4ed8,#0ea5e9); }
            .icon-kpi { background:linear-gradient(135deg,#0ea5e9,#3b82f6); } .icon-config { background:linear-gradient(135deg,#4f46e5,#7c3aed); }
            .icon-health { background:linear-gradient(135deg,#059669,#10b981); }
            .icon-alert::before{content:"!";} .icon-live::before{content:">";}
            .icon-kpi::before{content:"o";} .icon-config::before{content:"=";} .icon-health::before{content:"+";}
            .strong-box-title { border: 2px solid #93c5fd; border-radius: 10px; padding: 0.45rem 0.7rem; font-size: 1.05rem; font-weight: 800; margin-bottom: 0.45rem; color: #1e3a8a; background: #eff6ff; }
            .section-box-title { border: 2px solid #60a5fa; border-radius: 12px; padding: 0.5rem 0.8rem; font-size: 1.15rem; font-weight: 900; color: #1e40af; background: #eff6ff; margin: 0.35rem 0 0.7rem 0; }
            .input-box-title { border: 2px solid #93c5fd; border-radius: 10px; padding: 0.4rem 0.65rem; font-size: 1.0rem; font-weight: 800; color: #1e3a8a; background: #eff6ff; margin: 0.2rem 0 0.45rem 0; }
            .csv-help-card { border: 1px solid #bfdbfe; background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%); border-radius: 14px; padding: 0.75rem 0.9rem; margin: 0.25rem 0 0.75rem 0; box-shadow: 0 6px 18px rgba(37, 99, 235, 0.08); }
            .csv-help-title { font-size: 1rem; font-weight: 900; color: #1e3a8a; margin-bottom: 0.35rem; }
            .csv-help-line { font-size: 0.95rem; color: #1f2937; margin: 0.2rem 0; line-height: 1.35; }
            .csv-help-line code { background: #dbeafe; color: #1e3a8a; border-radius: 6px; padding: 0.08rem 0.32rem; font-weight: 700; }
            .csv-uploader-title { font-size: 1rem; font-weight: 900; color: #1e3a8a; margin: 0.2rem 0 0.45rem 0; }
            [data-testid="stFileUploader"] { background: #ffffff; border: 2px dashed #93c5fd; border-radius: 12px; padding: 0.65rem 0.65rem 0.25rem 0.65rem; }
            [data-testid="stFileUploader"]:hover { border-color: #60a5fa; background: #f8fbff; }
            button[data-baseweb="tab"] { font-weight: 800 !important; font-size: 1.05rem !important; }
            .stButton > button { background: linear-gradient(135deg, #2563eb, #0891b2) !important; color: #ffffff !important; border: none !important; font-weight: 700 !important; }
            .stButton > button:hover { background: linear-gradient(135deg, #1d4ed8, #0e7490) !important; }
            div[data-testid="stFormSubmitButton"] > button { background: linear-gradient(135deg, #2563eb, #0891b2) !important; color: #ffffff !important; border: none !important; font-weight: 700 !important; }
            section[data-testid="stSidebar"] { background:linear-gradient(180deg,#111827 0%,#1f2937 100%); }
            section[data-testid="stSidebar"] * { color:#e5e7eb !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_title(icon_class: str, text: str, sidebar: bool = False) -> None:
    html = f'<div class="title-row"><span class="icon {icon_class}"></span><span>{text}</span></div>'
    if sidebar:
        st.sidebar.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1 style="margin:0;">Centre de commandement du renseignement en cas de catastrophe</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _init_auth_state() -> None:
    if "is_authenticated" not in st.session_state:
        st.session_state["is_authenticated"] = False


def _render_login_screen() -> bool:
    logo_path = ROOT_DIR / "image" / "logo_au.png"
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] { display: none !important; }
            header[data-testid="stHeader"] { display: none !important; }
            .block-container { padding-top: 0rem !important; }
            .login-card { width: 100%; max-width: 620px; background: white; border: 1px solid #bfdbfe; border-radius: 22px; padding: 1.5rem; box-shadow: 0 18px 40px rgba(0,0,0,0.1); margin: auto; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)
        st.markdown(
            "<h2 style='text-align:center;'>Connexion Administrateur</h2>",
            unsafe_allow_html=True,
        )
        with st.form("auth_form"):
            password_input = st.text_input("Mot de passe", type="password")
            if st.form_submit_button("🔐 Se connecter", use_container_width=True):
                if password_input in ADMIN_PASSWORDS:
                    st.session_state["is_authenticated"] = True
                    st.rerun()
                else:
                    st.error("Mot de passe incorrect")
        st.markdown("</div>", unsafe_allow_html=True)
    return st.session_state.get("is_authenticated", False)


def call_api(payload: Dict[str, str]) -> Dict[str, Any]:
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, json=payload, timeout=60)
            if response.ok:
                return response.json()
            if response.status_code == 503 and attempt < max_retries - 1:
                time.sleep(5 + attempt * 5)
                continue
            response.raise_for_status()
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Erreur API : {e}")
            time.sleep(2)
    return {}


def plot_impact_words(impact_words: Dict[str, float], chart_key: str) -> None:
    if not impact_words:
        st.info("Aucune donnée d'importance disponible.")
        return
    df = pd.DataFrame(
        {"word": list(impact_words.keys()), "impact": list(impact_words.values())}
    ).sort_values("impact", ascending=False)
    fig = px.bar(
        df,
        x="word",
        y="impact",
        color="impact",
        color_continuous_scale="Tealrose",
        title="Importance des mots (BERT)",
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def plot_map(geo_coords: List[List[float]], map_key: str) -> None:
    center = geo_coords[0] if geo_coords else [20.0, 0.0]
    fmap = folium.Map(location=center, zoom_start=2, tiles="CartoDB dark_matter")
    for lat, lon in geo_coords:
        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            fill=True,
            color="#60a5fa",
            fill_color="#38bdf8",
            fill_opacity=0.9,
        ).add_to(fmap)
    st_folium(fmap, width=None, height=400, key=map_key)


def resolve_geo_coords(
    payload: Dict[str, str], pred: Dict[str, Any]
) -> List[List[float]]:
    geo_coords = pred.get("geo_coords", [])
    if geo_coords:
        return geo_coords
    location = (payload.get("location") or "").strip().lower()
    if location in KNOWN_LOCATION_COORDS:
        return [KNOWN_LOCATION_COORDS[location]]
    return []


def render_prediction_result(
    payload: Dict[str, str], pred: Dict[str, Any], context_key: str
) -> None:
    section_title("icon-kpi", "Résultat de l'analyse")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Info traduction
    if pred.get("detected_lang", "en") != "en":
        st.info(
            f"🌍 Langue détectée : `{pred['detected_lang']}`. Traduction automatique utilisée."
        )
        st.caption(f"Traduction : {pred.get('translated_text', '')}")

    c1, c2, c3 = st.columns(3)
    is_disaster = pred["is_disaster"]
    c1.metric(
        "Catastrophe",
        "OUI" if is_disaster else "NON",
        delta="Alerte" if is_disaster else None,
        delta_color="inverse",
    )
    c2.metric("Confiance", f"{pred['confidence'] * 100:.1f}%")
    c3.metric("Modèle", pred.get("model_name", "BERT"))
    style_metric_cards(border_color="#dbeafe")

    st.markdown(f"**Texte analysé (nettoyé)** : `{pred.get('clean_text', '')}`")
    st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns([1.2, 1])
    with left:
        plot_impact_words(pred.get("impact_words", {}), f"chart_{context_key}")
    with right:
        coords = resolve_geo_coords(payload, pred)
        plot_map(coords, f"map_{context_key}")


def manual_prediction() -> None:
    st.markdown(
        '<div class="section-box-title">Analyse individuelle</div>',
        unsafe_allow_html=True,
    )
    with st.form("manual_form"):
        text = st.text_area(
            "Tweet à analyser",
            height=100,
            placeholder="Ex: Huge wildfire spreading in California...",
        )
        col1, col2 = st.columns(2)
        keyword = col1.text_input("Mot-clé (optionnel)")
        location = col2.text_input("Lieu (optionnel)")
        submitted = st.form_submit_button(
            "Lancer l'intelligence BERT", use_container_width=True
        )

    if submitted and text:
        with st.spinner("Analyse BERT en cours..."):
            try:
                payload = {"text": text, "keyword": keyword, "location": location}
                result = call_api(payload)
                render_prediction_result(payload, result, "manual")
            except Exception as e:
                st.error(f"Erreur : {e}")


def main() -> None:
    setup_page()
    _init_auth_state()

    if not st.session_state["is_authenticated"]:
        _render_login_screen()
        return

    render_header()

    # Barre latérale
    with st.sidebar:
        section_title("icon-config", "Configuration", sidebar=True)
        st.info("Statut API : Connecté")
        if st.button("🔄 Rafraîchir le modèle MLflow", use_container_width=True):
            try:
                res = requests.post(REFRESH_URL, timeout=30).json()
                st.success(f"Modèle rechargé : {res.get('model_name')}")
                time.sleep(2)
                st.rerun()
            except Exception as e:
                st.error(f"Erreur refresh : {e}")

    tab1, tab2 = st.tabs(["📝 Manuel", "📊 Batch CSV"])
    with tab1:
        manual_prediction()
    with tab2:
        st.warning("Fonctionnalité batch en cours de mise à jour pour BERT.")


if __name__ == "__main__":
    main()
