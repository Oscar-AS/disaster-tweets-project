import sys
import time
import os
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.bootstrap import ensure_packages

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

import folium
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_folium import st_folium

API_URL = "https://disaster-tweets-project.onrender.com/predict"
ADMIN_PASSWORDS = {
    os.getenv("ADMIN_PASSWORD_1", "ADMIN1"),
    os.getenv("ADMIN_PASSWORD_2", "ADMIN2"),
    os.getenv("ADMIN_PASSWORD_3", "ADMIN3"),
}
SIMULATED_TWEETS = [
    {"keyword": "wildfire", "location": "Madrid", "text": "Massive wildfire spreading quickly near Madrid suburbs, evacuation underway!"},
    {"keyword": "sunset", "location": "Paris", "text": "Beautiful sunset in Paris, everyone is enjoying the evening."},
    {"keyword": "flood", "location": "Lagos", "text": "Flood warnings issued in Lagos after heavy overnight rain."},
    {"keyword": "explosion", "location": "New York", "text": "Explosion reported downtown New York, emergency teams dispatched."},
]
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
            .icon-health { background:linear-gradient(135deg,#059669,#10b981); } .icon-alert::before{content:"!";} .icon-live::before{content:">";}
            .icon-kpi::before{content:"o";} .icon-config::before{content:"=";} .icon-health::before{content:"+";}
            .strong-box-title {
                border: 2px solid #93c5fd;
                border-radius: 10px;
                padding: 0.45rem 0.7rem;
                font-size: 1.05rem;
                font-weight: 800;
                margin-bottom: 0.45rem;
                color: #1e3a8a;
                background: #eff6ff;
            }
            .section-box-title {
                border: 2px solid #60a5fa;
                border-radius: 12px;
                padding: 0.5rem 0.8rem;
                font-size: 1.15rem;
                font-weight: 900;
                color: #1e40af;
                background: #eff6ff;
                margin: 0.35rem 0 0.7rem 0;
            }
            .input-box-title {
                border: 2px solid #93c5fd;
                border-radius: 10px;
                padding: 0.4rem 0.65rem;
                font-size: 1.0rem;
                font-weight: 800;
                color: #1e3a8a;
                background: #eff6ff;
                margin: 0.2rem 0 0.45rem 0;
            }
            button[data-baseweb="tab"] {
                font-weight: 800 !important;
                font-size: 1.05rem !important;
            }
            .stButton > button {
                background: linear-gradient(135deg, #2563eb, #0891b2) !important;
                color: #ffffff !important;
                border: none !important;
                font-weight: 700 !important;
            }
            .stButton > button:hover {
                background: linear-gradient(135deg, #1d4ed8, #0e7490) !important;
            }
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
    if "auth_error" not in st.session_state:
        st.session_state["auth_error"] = ""


def _render_login_screen() -> bool:
    logo_path = ROOT_DIR / "image" / "logo_au.png"
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] { display: none !important; }
            header[data-testid="stHeader"] { display: none !important; }
            div[data-testid="stToolbar"] { display: none !important; }
            div[data-testid="stDecoration"] { display: none !important; }
    
            div[data-testid="stSidebarNav"] { display: none !important; }
            div[data-testid="stPageNav"] { display: none !important; }
            #MainMenu { visibility: hidden !important; }
            .block-container { padding-top: 0rem !important; }
            /* Masquage chirurgical de la barre parasite du haut
               sans impacter le champ password d'authentification. */
            }
            .login-shell {
                min-height: auto;
                display: flex;
                align-items: flex-start;
                padding-top: 0;
            }
            .login-card {
                width: 100%;
                max-width: 620px;
                background: linear-gradient(145deg, rgba(255,255,255,0.98) 0%, rgba(239,246,255,0.96) 100%);
                border: 1px solid #bfdbfe;
                border-radius: 22px;
                padding: 1.1rem 1.3rem 1.1rem 1.3rem;
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.14);
            }
            .login-title {
                font-size: 1.5rem;
                font-weight: 900;
                color: #1e40af;
                margin-bottom: 0.3rem;
                text-align: center;
            }
            .login-subtitle {
                font-size: 0.98rem;
                color: #334155;
                text-align: center;
                margin-bottom: 0.95rem;
                font-weight: 800;
            }
            .auth-hint {
                font-weight: 800;
                color: #1e3a8a;
                text-align: center;
                margin-top: 0.6rem;
                font-size: 0.85rem;
            }
            .auth-msg {
                border: 1px solid #93c5fd;
                background: #eff6ff;
                color: #1e40af;
                border-radius: 10px;
                padding: 0.55rem 0.7rem;
                margin-top: 0.55rem;
                margin-bottom: 0.2rem;
                font-weight: 800;
                text-align: center;
            }
            div[data-testid="stForm"] button[kind="primary"] {
                background: linear-gradient(135deg, #16a34a, #22c55e) !important;
                color: #ffffff !important;
                border: none !important;
                font-weight: 800 !important;
            }
            div[data-testid="stForm"] button[kind="primary"]:hover {
                background: linear-gradient(135deg, #15803d, #16a34a) !important;
                color: #ffffff !important;
            }
            div[data-testid="stFormSubmitButton"] > button {
                background: linear-gradient(135deg, #16a34a, #22c55e) !important;
                color: #ffffff !important;
                border: none !important;
                font-weight: 800 !important;
            }
            div[data-testid="stFormSubmitButton"] > button:hover {
                background: linear-gradient(135deg, #15803d, #16a34a) !important;
                color: #ffffff !important;
            }
            .login-card div[data-testid="stForm"] div[data-testid="stTextInput"] input {
                border: 2px solid #93c5fd !important;
                border-radius: 10px !important;
                background: #ffffff !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="login-shell">', unsafe_allow_html=True)
    left, center, right = st.columns([1.2, 2.4, 1.2])
    with center:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        if logo_path.exists():
            l1, l2, l3 = st.columns([1.1, 1.6, 1.1])
            with l2:
                st.image(str(logo_path), width="stretch")
        st.markdown('<div class="login-title"><strong>Connexion Administrateur</strong></div>', unsafe_allow_html=True)
        with st.form("auth_form", clear_on_submit=False):
            st.markdown("**Mot de passe administrateur**")
            password_input = st.text_input("Mot de passe administrateur", type="password", key="admin_password_input", label_visibility="collapsed", placeholder="Entrez votre mot de passe")
            submit_login = st.form_submit_button("🔐 Se connecter", width="stretch", type="primary")
        if submit_login:
            if password_input in ADMIN_PASSWORDS:
                st.session_state["is_authenticated"] = True
                st.session_state["auth_error"] = ""
                st.rerun()
            else:
                st.session_state["auth_error"] = "Mot de passe invalide. Veuillez reessayer."
        if st.session_state.get("auth_error"):
            st.markdown(f'<div class="auth-msg"><strong>{st.session_state["auth_error"]}</strong></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    return st.session_state.get("is_authenticated", False)


def call_api(payload: Dict[str, str]) -> Dict[str, Any]:
    response = requests.post(API_URL, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def plot_impact_words(impact_words: Dict[str, float]) -> None:
    if not impact_words:
        st.info("Aucune contribution de mot disponible pour ce tweet.")
        return
    df = pd.DataFrame({"word": list(impact_words.keys()), "impact": list(impact_words.values())}).sort_values("impact", ascending=False)
    fig = px.bar(df, x="word", y="impact", color="impact", color_continuous_scale="Tealrose", title="Influence de chaque mot")
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")


def plot_map(geo_coords: List[List[float]], map_key: str) -> None:
    center = geo_coords[0] if geo_coords else [20.0, 0.0]
    fmap = folium.Map(location=center, zoom_start=2, tiles="CartoDB dark_matter")
    for lat, lon in geo_coords:
        folium.CircleMarker(location=[lat, lon], radius=7, fill=True, color="#60a5fa", fill_color="#38bdf8", fill_opacity=0.9).add_to(fmap)
    # Cle stable par contexte (manual/live) pour eviter le clignotement.
    st_folium(fmap, width=None, height=420, key=map_key)


def resolve_geo_coords(payload: Dict[str, str], pred: Dict[str, Any]) -> List[List[float]]:
    geo_coords = pred.get("geo_coords", [])
    if geo_coords:
        return geo_coords
    location = (payload.get("location") or "").strip().lower()
    if location in KNOWN_LOCATION_COORDS:
        return [KNOWN_LOCATION_COORDS[location]]
    return []


def render_prediction_result(payload: Dict[str, str], pred: Dict[str, Any], context_key: str) -> None:
    section_title("icon-kpi", "Resultat de prediction")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"**Mot-cle**: `{payload.get('keyword', '') or 'N/A'}`")
    st.markdown(f"**Emplacement**: `{payload.get('location', '') or 'N/A'}`")
    st.markdown(f"**Tweet analyse**: {payload.get('text', '')}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Catastrophe", "Oui" if pred["is_disaster"] else "Non")
    # Compatibilite avec plusieurs schemas d'API:
    # - schema actuel de ton projet: `score`, `impact_words`, `geo_coords`
    # - schema de l'API deployee: `confidence` (et souvent pas d'impact/geo)
    pred_score = pred.get("score", pred.get("confidence", 0.0))
    try:
        pred_score_f = float(pred_score)
    except Exception:
        pred_score_f = 0.0

    c2.metric("Score de risque", f"{pred_score_f:.4f}")
    resolved_geo_coords = resolve_geo_coords(payload, pred)
    c3.metric("Lieux detectes", f"{len(resolved_geo_coords)}")
    style_metric_cards(border_color="#dbeafe", background_color="#ffffff")

    # Infos complementaires renvoyees par certaines APIs (ex: backend Render).
    model_name = pred.get("model_name")
    if model_name:
        st.markdown(f"**Modele utilise**: `{model_name}`")
    clean_text = pred.get("clean_text")
    if clean_text:
        st.caption(f"Texte nettoye: {clean_text}")

    st.markdown("</div>", unsafe_allow_html=True)
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        plot_impact_words(pred.get("impact_words", {}))
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        plot_map(resolved_geo_coords, map_key=f"folium_map_{context_key}")
        st.markdown("</div>", unsafe_allow_html=True)


def manual_prediction(default_payload: Dict[str, str]) -> None:
    st.markdown('<div class="section-box-title"><strong>Analyse manuelle</strong></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="strong-box-title"><strong>Mot-cle</strong></div>', unsafe_allow_html=True)
        keyword = st.text_input("Mot-cle", value=default_payload.get("keyword", ""), key="manual_keyword", label_visibility="collapsed")
    with c2:
        st.markdown('<div class="strong-box-title"><strong>Emplacement</strong></div>', unsafe_allow_html=True)
        location = st.text_input("Emplacement", value=default_payload.get("location", ""), key="manual_location", label_visibility="collapsed")
    st.markdown('<div class="input-box-title"><strong>Entrez un tweet a analyser</strong></div>', unsafe_allow_html=True)
    text = st.text_area("Entrez un tweet a analyser", value=default_payload.get("text", ""), height=110, key="manual_text", label_visibility="collapsed")
    current_payload = {"keyword": keyword, "location": location, "text": text}
    if st.button("Analyser le tweet", width="stretch", type="primary", key="manual_predict_button"):
        try:
            pred = call_api(current_payload)
            st.session_state["manual_last_payload"] = current_payload
            st.session_state["manual_last_pred"] = pred
        except Exception as exc:
            st.error(f"Echec appel API: {exc}")

    # Affiche le dernier resultat connu, seulement si le formulaire
    # correspond encore au tweet analyse (evite l'impression de texte fige).
    if "manual_last_payload" in st.session_state and "manual_last_pred" in st.session_state:
        if st.session_state["manual_last_payload"] == current_payload:
            render_prediction_result(st.session_state["manual_last_payload"], st.session_state["manual_last_pred"], context_key="manual")
        else:
            st.info("Le texte a change. Clique sur 'Analyser le tweet' pour mettre a jour le resultat.")


def simulate_stream(delay_seconds: float) -> None:
    st.markdown('<div class="section-box-title"><strong>Simulation de mode en direct</strong></div>', unsafe_allow_html=True)
    placeholder = st.empty()
    start_live = st.button("Demarrer la simulation", width="stretch", key="live_start_button")
    if start_live:
        for i, payload in enumerate(SIMULATED_TWEETS, start=1):
            with placeholder.container():
                st.info(f"Tweet {i}/{len(SIMULATED_TWEETS)} en cours d'analyse...")
                try:
                    pred = call_api(payload)
                    st.session_state["live_last_payload"] = payload
                    st.session_state["live_last_pred"] = pred
                    render_prediction_result(payload, pred, context_key=f"live_{i}")
                except Exception as exc:
                    st.error(f"Erreur API: {exc}")
            time.sleep(delay_seconds)

    # Garde visible le dernier resultat de la simulation.
    # Important: ne pas l'afficher dans le meme rerun que le bouton,
    # sinon certains composants (st_folium) peuvent avoir des cles dupliquees.
    if (not start_live) and "live_last_payload" in st.session_state and "live_last_pred" in st.session_state:
        render_prediction_result(st.session_state["live_last_payload"], st.session_state["live_last_pred"], context_key="live_last")


def sidebar_controls() -> Dict[str, Any]:
    logo_path = ROOT_DIR / "image" / "logo.png"
    if logo_path.exists():
        _, center, _ = st.sidebar.columns([1, 2, 1])
        with center:
            st.image(str(logo_path), width="stretch")
        st.sidebar.write("")
    if st.sidebar.button("Deconnexion", width="stretch", key="logout_button"):
        st.session_state["is_authenticated"] = False
        st.rerun()
    section_title("icon-config", "Configuration", sidebar=True)
    api_url = st.sidebar.text_input("URL API", value=API_URL, key="sidebar_api_url")
    delay = st.sidebar.slider("Delai simulation (secondes)", 0.5, 5.0, 1.5, 0.5, key="sidebar_delay")
    starter = st.sidebar.selectbox(
        "Exemple de tweet initial",
        options=SIMULATED_TWEETS,
        index=0,
        format_func=lambda item: f"{item['keyword']} - {item['location']}: {item['text'][:36]}...",
        key="sidebar_starter",
    )
    section_title("icon-health", "Etat systeme", sidebar=True)
    try:
        normalized_api_url = api_url.strip().rstrip("/")
        if normalized_api_url.endswith("/predict"):
            health_url = f"{normalized_api_url[:-len('/predict')]}/health"
        else:
            health_url = f"{normalized_api_url}/health"
        health = requests.get(health_url, timeout=12)
        if health.ok:
            st.sidebar.success("API connectee")
        else:
            st.sidebar.warning("API repond avec un statut inattendu")
    except Exception as exc:
        st.sidebar.error(f"API non joignable ({type(exc).__name__})")
    return {"api_url": api_url, "delay": delay, "starter": starter}


def main() -> None:
    setup_page()
    _init_auth_state()
    if not st.session_state.get("is_authenticated", False):
        _render_login_screen()
        return

    controls = sidebar_controls()
    global API_URL
    API_URL = controls["api_url"]
    render_header()
    tab_manual, tab_live = st.tabs(["Analyse manuelle", "Simulation de mode en direct"])
    with tab_manual:
        manual_prediction(controls["starter"])
    with tab_live:
        simulate_stream(controls["delay"])


if __name__ == "__main__":
    main()

