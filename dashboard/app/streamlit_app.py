"""
================================================================================
DISASTER INTELLIGENCE — Guide pour un lecteur NON technique
================================================================================
Ce fichier pilote toute l'application web (tableau de bord) que vous voyez
dans le navigateur. En résumé :

  1) Connexion — un mot de passe limite l'accès au tableau de bord.
  2) Analyse manuelle — vous saisissez un tweet ; le programme l'envoie à une
     API (service sur Internet) qui estime s'il s'agit d'un message lié à une
     catastrophe, avec un score de confiance.
  3) Analyse par lot — vous téléversez un fichier CSV ; chaque ligne est
     envoyée à la même API ; vous pouvez télécharger un fichier de résultats.
  4) Carte et graphiques — affichés ici, à partir des réponses de l'API ou
     d'explications locales simplifiées si l'API ne renvoie pas tout.

Vocabulaire utile :
  • « API » = programme distant auquel on envoie du texte et qui renvoie une
    prédiction (un peu comme poser une question à un serveur).
  • « Streamlit » = outil qui transforme ce script Python en pages web.
  • « Session » = mémoire temporaire du navigateur pour garder login, résultats.
================================================================================
"""

# --- Etape 1 : outils Python de base (bibliothèques / "boîtes à outils") ---
import sys  # Chemins d'exécution et accès au système
import time  # Attentes entre deux essais si l'API répond mal (cold start)
import os  # Lecture optionnelle de mots de passe via variables d'environnement
import re  # Recherche de mots dans le texte (pour graphique d'impact local)
from pathlib import Path  # Manipulation propre des dossiers et fichiers
from typing import Any, Dict, List  # Étiquettes pour clarifier le type des données

# Dossier racine du projet (le dossier parent du dossier `app/`)
ROOT_DIR = Path(__file__).resolve().parents[1]
# Permet d'importer `src.bootstrap` même si on lance le script depuis un autre endroit
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.bootstrap import ensure_packages

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
        ("deep-translator", "deep_translator"),
        ("langdetect", "langdetect"),
    ]
)

# --- Etape 2 : outils pour carte, tableaux, graphiques, web ---
import folium  # Carte du monde (points géographiques)
import pandas as pd  # Tableaux de données (comme un petit Excel en mémoire)
import plotly.express as px  # Graphiques en barres interactifs
import requests  # Appels HTTP vers l'API de prédiction et /health
import streamlit as st  # Tout l'affichage web : boutons, onglets, formulaires
from streamlit_extras.metric_cards import style_metric_cards  # Mise en forme des encarts chiffrés
from streamlit_folium import st_folium  # Affichage de la carte Folium dans Streamlit

# Adresse par défaut du service de prédiction (modifiable dans la barre latérale)
API_URL = "https://disaster-tweets-project.onrender.com/predict"
# Trois mots de passe acceptés ; on peut les remplacer par des variables d'environnement
ADMIN_PASSWORDS = {
    os.getenv("ADMIN_PASSWORD_1", "ADMIN1"),
    os.getenv("ADMIN_PASSWORD_2", "ADMIN2"),
    os.getenv("ADMIN_PASSWORD_3", "ADMIN3"),
}
# Villes connues : si l'API ne renvoie pas de coordonnées, on place un point sur la carte ici
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
# Poids "simples" pour le graphique d'impact si l'API ne renvoie pas de détail par mot
LOCAL_IMPACT_WEIGHTS = {
    "earthquake": 0.22,
    "flood": 0.2,
    "wildfire": 0.24,
    "fire": 0.18,
    "hurricane": 0.22,
    "evacuation": 0.18,
    "disaster": 0.16,
    "collapsed": 0.14,
    "injured": 0.14,
    "dead": 0.2,
    "tsunami": 0.24,
    "explosion": 0.22,
    "rescue": 0.12,
    "storm": 0.16,
    "warning": 0.1,
    "emergency": 0.12,
}


def setup_page() -> None:
    """
    Pour un non-technicien : configure la page (titre dans l'onglet du navigateur,
    largeur) et injecte du « CSS » = règles de couleurs et de forme pour que
    l'interface soit lisible et cohérente (cartes, boutons, barre latérale).
    """
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
            .csv-help-card {
                border: 1px solid #bfdbfe;
                background: linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
                border-radius: 14px;
                padding: 0.75rem 0.9rem;
                margin: 0.25rem 0 0.75rem 0;
                box-shadow: 0 6px 18px rgba(37, 99, 235, 0.08);
            }
            .csv-help-title {
                font-size: 1rem;
                font-weight: 900;
                color: #1e3a8a;
                margin-bottom: 0.35rem;
            }
            .csv-help-line {
                font-size: 0.95rem;
                color: #1f2937;
                margin: 0.2rem 0;
                line-height: 1.35;
            }
            .csv-help-line code {
                background: #dbeafe;
                color: #1e3a8a;
                border-radius: 6px;
                padding: 0.08rem 0.32rem;
                font-weight: 700;
            }
            .csv-uploader-title {
                font-size: 1rem;
                font-weight: 900;
                color: #1e3a8a;
                margin: 0.2rem 0 0.45rem 0;
            }
            [data-testid="stFileUploader"] {
                background: #ffffff;
                border: 2px dashed #93c5fd;
                border-radius: 12px;
                padding: 0.65rem 0.65rem 0.25rem 0.65rem;
            }
            [data-testid="stFileUploader"]:hover {
                border-color: #60a5fa;
                background: #f8fbff;
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
            div[data-testid="stFormSubmitButton"] > button {
                background: linear-gradient(135deg, #2563eb, #0891b2) !important;
                color: #ffffff !important;
                border: none !important;
                font-weight: 700 !important;
            }
            div[data-testid="stFormSubmitButton"] > button:hover {
                background: linear-gradient(135deg, #1d4ed8, #0e7490) !important;
                color: #ffffff !important;
            }
            section[data-testid="stSidebar"] { background:linear-gradient(180deg,#111827 0%,#1f2937 100%); }
            section[data-testid="stSidebar"] * { color:#e5e7eb !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_title(icon_class: str, text: str, sidebar: bool = False) -> None:
    """Affiche un petit titre avec icône colorée, soit au centre soit dans la colonne de gauche."""
    html = f'<div class="title-row"><span class="icon {icon_class}"></span><span>{text}</span></div>'
    if sidebar:
        st.sidebar.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


def render_header() -> None:
    """Bandeau bleu tout en haut du tableau de bord avec le titre principal."""
    st.markdown(
        """
        <div class="hero">
            <h1 style="margin:0;">Centre de commandement du renseignement en cas de catastrophe</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _init_auth_state() -> None:
    """Prépare la mémoire Streamlit : est-ce que l'utilisateur est déjà connecté ? Message d'erreur éventuel."""
    if "is_authenticated" not in st.session_state:
        st.session_state["is_authenticated"] = False
    if "auth_error" not in st.session_state:
        st.session_state["auth_error"] = ""


def _render_login_screen() -> bool:
    """
    Écran de connexion uniquement : logo, champ mot de passe, bouton.
    Tant que le mot de passe n'est pas bon, le reste du tableau de bord ne s'affiche pas.
    """
    logo_path = ROOT_DIR / "image" / "logo_au.png"
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] { display: none !important; }
            header[data-testid="stHeader"] { display: none !important; }
            div[data-testid="stToolbar"] { display: none !important; }
            div[data-testid="stDecoration"] { display: none !important; }

            /* Barre latérale et menus Streamlit masqués ici pour un écran de connexion épuré */
            div[data-testid="stSidebarNav"] { display: none !important; }
            div[data-testid="stPageNav"] { display: none !important; }
            #MainMenu { visibility: hidden !important; }
            .block-container { padding-top: 0rem !important; }
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


def health_url_from_predict_url(api_url: str) -> str:
    """URL `/health` dérivée de l’URL `/predict` (ou base seule) saisie dans la barre latérale."""
    u = api_url.strip().rstrip("/")
    if u.endswith("/predict"):
        return f"{u[:-len('/predict')]}/health"
    return f"{u}/health"


def _api_error_detail(response: requests.Response) -> str:
    """Extrait un message lisible depuis une réponse JSON FastAPI (`detail`) ou le corps brut."""
    try:
        payload = response.json()
        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()
        if isinstance(detail, list) and detail:
            first = detail[0]
            if isinstance(first, dict) and first.get("msg"):
                return str(first["msg"])
            return str(first)
    except Exception:
        pass
    text = (response.text or "").strip()
    return text[:800] if text else f"HTTP {response.status_code}"


def call_api(payload: Dict[str, str]) -> Dict[str, Any]:
    """
    Pour un non-technicien : envoie le tweet (mot-clé, lieu, texte) au serveur de prédiction.
    Si le serveur « dort » (hébergement gratuit) ou est saturé, on réessaie plusieurs fois
    au lieu d'afficher une erreur tout de suite.
    """
    health_url = health_url_from_predict_url(API_URL)
    max_retries = 8
    last_response: requests.Response | None = None
    has_warmed_up = False

    # Prévol : uniquement si l'API expose `predict_ready` (version récente). Sinon on tente /predict
    # (anciennes images Render : /health sans ce champ — ne pas bloquer à tort).
    try:
        health = requests.get(health_url, timeout=35)
        if health.ok:
            body = health.json()
            if "predict_ready" in body and body.get("predict_ready") is False:
                raise RuntimeError(
                    "Côté serveur, aucune prédiction n'est autorisée pour l'instant : le modèle MLflow "
                    "n'est pas chargé et le repli heuristique est désactivé (ALLOW_HEURISTIC_FALLBACK). "
                    "Sur Render : mettez ALLOW_HEURISTIC_FALLBACK=true ou configurez DAGSHUB_USER_TOKEN "
                    "+ MLFLOW_TRACKING_USERNAME, puis redéployez. "
                    f"Réponse /health : {body}"
                )
    except RuntimeError:
        raise
    except Exception:
        pass  # Cold start : /health peut encore échouer, on tente quand même POST.

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, json=payload, timeout=120)
            last_response = response
            if response.status_code in (502, 503, 504) and attempt < max_retries - 1:
                if response.status_code == 503 and not has_warmed_up:
                    try:
                        requests.get(health_url, timeout=45)
                    except Exception:
                        pass
                    has_warmed_up = True
                time.sleep(6 + attempt * 8)
                continue
            if response.ok:
                return response.json()
            if response.status_code == 503:
                raise RuntimeError(
                    "Erreur serveur 503 — "
                    + _api_error_detail(response)
                    + " Si le texte indique MLflow ou le modèle, corrigez le déploiement Render ; sinon réessayez après le cold start."
                )
            detail = _api_error_detail(response)
            raise RuntimeError(f"Erreur API {response.status_code}: {detail}")
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < max_retries - 1:
                time.sleep(6 + attempt * 8)
                continue
            raise

    if last_response is not None:
        if last_response.status_code == 503:
            raise RuntimeError(
                "Erreur serveur 503 après plusieurs tentatives — "
                + _api_error_detail(last_response)
                + " Utilisez « Réveiller l'API » puis réessayez, ou vérifiez que le modèle est bien chargé côté Render."
            )
        if not last_response.ok:
            raise RuntimeError(f"Erreur API {last_response.status_code}: {_api_error_detail(last_response)}")
        return last_response.json()
    raise RuntimeError("API indisponible apres plusieurs tentatives.")


def call_api_with_visible_feedback(payload: Dict[str, str], running_label: str) -> Dict[str, Any]:
    """
    Comme `call_api`, mais avec une bannière de progression Streamlit : l'utilisateur voit
    clairement que l'analyse est en cours jusqu'à la réponse du serveur.
    """
    with st.status(running_label, expanded=True) as status:
        result = call_api(payload)
        status.update(label="Traitement termine", state="complete", expanded=False)
    return result


def translate_preview_if_needed(text: str) -> Dict[str, str]:
    """
    Pré-visualise la traduction côté appli :
    - si le texte est déjà anglais (ou très court), on le garde
    - sinon on traduit en anglais avant envoi à l'API
    """
    out = {"text_for_prediction": text, "detected_lang": "unknown", "translated": "false"}
    stripped = (text or "").strip()
    if not stripped:
        return out

    min_len = int(os.getenv("TRANSLATE_MIN_CHARS", "8"))
    if len(stripped) < min_len:
        return out

    try:
        from langdetect import LangDetectException, detect  # type: ignore[import-untyped]

        try:
            lang = (detect(stripped) or "").lower()
        except LangDetectException:
            return out
        if not lang:
            return out
        out["detected_lang"] = lang
        if lang in {"en", "eng"}:
            return out
    except Exception:
        return out

    try:
        from deep_translator import GoogleTranslator  # type: ignore[import-untyped]

        translated = GoogleTranslator(source=out["detected_lang"], target="en").translate(stripped)
        if isinstance(translated, str) and translated.strip():
            out["text_for_prediction"] = translated.strip()
            out["translated"] = "true"
    except Exception:
        return out
    return out


def plot_impact_words(impact_words: Dict[str, float], chart_key: str) -> None:
    """Diagramme en barres : chaque mot et son « poids » dans la décision (ou estimation locale)."""
    if not impact_words:
        st.info("Aucune contribution de mot disponible pour ce tweet.")
        return
    df = pd.DataFrame({"word": list(impact_words.keys()), "impact": list(impact_words.values())}).sort_values("impact", ascending=False)
    fig = px.bar(df, x="word", y="impact", color="impact", color_continuous_scale="Tealrose", title="Influence de chaque mot")
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", key=chart_key)


def compute_local_impact_words(payload: Dict[str, str]) -> Dict[str, float]:
    """
    Si l'API ne renvoie pas la liste des mots « importants », ce bloc regarde le texte ici
    et attribue des petits scores aux mots d'alerte connus (feu, inondation...) pour quand
    même afficher un graphique compréhensible.
    """
    raw_text = " ".join(
        part for part in [payload.get("keyword", ""), payload.get("location", ""), payload.get("text", "")] if part
    ).lower()
    tokens = [tok for tok in re.findall(r"[a-zA-Z']+", raw_text) if tok]

    impacts: Dict[str, float] = {}
    for tok in tokens[:20]:
        if tok in LOCAL_IMPACT_WEIGHTS:
            impacts[tok] = max(impacts.get(tok, 0.0), LOCAL_IMPACT_WEIGHTS[tok])

    # Petit fallback: si aucun mot "risque" detecte, on montre les mots principaux neutres.
    if not impacts:
        unique_tokens: List[str] = []
        for tok in tokens:
            if tok not in unique_tokens and len(tok) > 3:
                unique_tokens.append(tok)
            if len(unique_tokens) >= 5:
                break
        for tok in unique_tokens:
            impacts[tok] = 0.02

    return dict(sorted(impacts.items(), key=lambda item: item[1], reverse=True))


def plot_map(geo_coords: List[List[float]], map_key: str) -> None:
    """Carte mondiale avec marqueurs aux coordonnées reçues ; sans coordonnées, vue par défaut centrée."""
    center = geo_coords[0] if geo_coords else [20.0, 0.0]
    fmap = folium.Map(location=center, zoom_start=2, tiles="CartoDB dark_matter")
    for lat, lon in geo_coords:
        folium.CircleMarker(location=[lat, lon], radius=7, fill=True, color="#60a5fa", fill_color="#38bdf8", fill_opacity=0.9).add_to(fmap)
    # Cle stable par contexte (manual/live) pour eviter le clignotement.
    st_folium(fmap, width=None, height=420, key=map_key)


def resolve_geo_coords(payload: Dict[str, str], pred: Dict[str, Any]) -> List[List[float]]:
    """Choisit où placer la carte : d'abord ce que l'API envoie, sinon une ville du dictionnaire connu."""
    geo_coords = pred.get("geo_coords", [])
    if geo_coords:
        return geo_coords
    location = (payload.get("location") or "").strip().lower()
    if location in KNOWN_LOCATION_COORDS:
        return [KNOWN_LOCATION_COORDS[location]]
    return []


def render_prediction_result(payload: Dict[str, str], pred: Dict[str, Any], context_key: str) -> None:
    """Affiche tout le bloc résultat : métriques Oui/Non, score, graphique des mots, carte."""
    section_title("icon-kpi", "Resultat de prediction")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"**Mot-cle**: `{payload.get('keyword', '') or 'N/A'}`")
    st.markdown(f"**Emplacement**: `{payload.get('location', '') or 'N/A'}`")
    analyzed_text = pred.get("clean_text") or payload.get("text", "")
    st.markdown(f"**Tweet analyse**: {analyzed_text}")
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
    impact_words = pred.get("impact_words", {})
    if not impact_words:
        impact_words = compute_local_impact_words(payload)

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
        plot_impact_words(impact_words, chart_key=f"impact_chart_{context_key}")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        plot_map(resolved_geo_coords, map_key=f"folium_map_{context_key}")
        st.markdown("</div>", unsafe_allow_html=True)


def manual_prediction() -> None:
    """
    Onglet « un tweet à la fois » : formulaire, envoi API, mémorisation du dernier résultat,
    message si l'utilisateur modifie le texte sans recliquer sur Analyser.
    """
    st.markdown('<div class="section-box-title"><strong>📝 Analyse manuelle</strong></div>', unsafe_allow_html=True)
    with st.form("manual_prediction_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="strong-box-title"><strong>Mot-cle</strong></div>', unsafe_allow_html=True)
            keyword = st.text_input("Mot-cle", key="manual_keyword", label_visibility="collapsed")
        with c2:
            st.markdown('<div class="strong-box-title"><strong>Emplacement</strong></div>', unsafe_allow_html=True)
            location = st.text_input("Emplacement", key="manual_location", label_visibility="collapsed")
        st.markdown('<div class="input-box-title"><strong>Entrez un tweet a analyser</strong></div>', unsafe_allow_html=True)
        text = st.text_area("Entrez un tweet a analyser", height=110, key="manual_text", label_visibility="collapsed")
        submit_manual = st.form_submit_button("Analyser le tweet", width="stretch", type="primary")

    current_payload = {"keyword": keyword, "location": location, "text": text}
    if submit_manual:
        try:
            translation_meta = translate_preview_if_needed(current_payload.get("text", ""))
            payload_for_api = dict(current_payload)
            payload_for_api["text"] = translation_meta["text_for_prediction"]

            st.session_state["manual_last_translation"] = translation_meta
            pred = call_api_with_visible_feedback(payload_for_api, "Analyse du tweet en cours...")
            st.session_state["manual_last_payload"] = dict(payload_for_api)
            st.session_state["manual_last_pred"] = pred
            st.session_state["manual_result_version"] = st.session_state.get("manual_result_version", 0) + 1
        except Exception as exc:
            st.error(f"Echec appel API: {exc}")

    if "manual_last_payload" in st.session_state and "manual_last_pred" in st.session_state:
        translation_meta = st.session_state.get("manual_last_translation", {})
        if translation_meta.get("translated") == "true":
            st.info("Traduction detectee avant prediction")
            st.caption(f"Langue detectee: `{translation_meta.get('detected_lang', 'unknown')}`")
            st.text_area(
                "Texte traduit (envoye au modele)",
                value=translation_meta.get("text_for_prediction", ""),
                height=100,
                key=f"manual_translation_preview_{st.session_state.get('manual_result_version', 0)}",
            )
        render_prediction_result(
            st.session_state["manual_last_payload"],
            st.session_state["manual_last_pred"],
            context_key=f"manual_{st.session_state.get('manual_result_version', 0)}",
        )
        if st.session_state["manual_last_payload"] != current_payload:
            st.info("Le texte a change. Clique sur 'Analyser le tweet' pour mettre a jour le resultat.")


def _csv_column_lookup(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Trouve la vraie colonne du fichier même si l'orthographe varie (text vs Texte, etc.)."""
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for name in candidates:
        if name in lower_map:
            return lower_map[name]
    return None


def _prepare_batch_dataframe(uploaded_file: Any) -> pd.DataFrame:
    """Lit le fichier téléversé et le transforme en tableau exploitable (CSV uniquement ici)."""
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    raise ValueError("Fichier non supporte: utilisez un fichier .csv")


def batch_csv_prediction(max_rows: int) -> None:
    """
    Onglet fichier : import CSV, détection automatique des colonnes, boucle ligne par ligne
    vers l'API, tableau des résultats et bouton de téléchargement.
    """
    st.markdown('<div class="section-box-title"><strong>📊 Analyse par lot (CSV)</strong></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="csv-help-card">
            <div class="csv-help-title">📄 Colonnes attendues (format flexible)</div>
            <div class="csv-help-line">✅ <strong>Obligatoire</strong> : une colonne <code>text</code>, <code>tweet</code>, <code>message</code>, <code>content</code> ou <code>texte</code></div>
            <div class="csv-help-line">➕ <strong>Optionnel</strong> : <code>keyword</code> / <code>mot_cle</code>, <code>location</code> / <code>lieu</code> / <code>place</code></div>
            <div class="csv-help-line">⚙️ Chaque ligne est envoyée à <code>/predict</code>, puis regroupée dans un tableau exportable.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="csv-uploader-title">⬆️ Charger un fichier CSV</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Charger un fichier CSV",
        type=["csv"],
        key="batch_csv_upload",
        label_visibility="collapsed",
        help="Fichier CSV uniquement (colonnes flexibles).",
    )

    if uploaded is None:
        st.session_state.pop("batch_results_df", None)
        st.session_state.pop("batch_upload_fingerprint", None)
        return

    fp = f"{getattr(uploaded, 'name', '')}_{getattr(uploaded, 'size', 0)}"
    if st.session_state.get("batch_upload_fingerprint") != fp:
        st.session_state["batch_upload_fingerprint"] = fp
        st.session_state.pop("batch_results_df", None)

    try:
        df = _prepare_batch_dataframe(uploaded)
    except Exception as exc:
        st.error(f"Lecture CSV impossible: {exc}")
        return

    text_col = _csv_column_lookup(df, ("text", "tweet", "message", "content", "texte"))
    if not text_col:
        st.error(
            "Aucune colonne texte trouvee. Ajoutez une colonne nommee par exemple : **text**, **tweet**, **message** ou **content**."
        )
        return

    kw_col = _csv_column_lookup(df, ("keyword", "mot-cle", "mot_cle", "key_word"))
    loc_col = _csv_column_lookup(df, ("location", "lieu", "place", "emplacement", "city"))

    n_total = min(len(df), max_rows)
    if len(df) > max_rows:
        st.warning(f"Le fichier contient {len(df)} lignes. Seules les **{max_rows}** premieres seront traitees.")
        df_work = df.head(max_rows).copy()
    else:
        df_work = df.copy()

    if st.button("Lancer l'analyse par lot", width="stretch", type="primary", key="batch_run_button"):
        results: List[Dict[str, Any]] = []
        progress = st.progress(0, text="Preparation...")
        status_box = st.empty()

        for i, (_, row) in enumerate(df_work.iterrows()):
            raw_text = row.get(text_col, "")
            text_val = "" if pd.isna(raw_text) else str(raw_text).strip()
            keyword_val = ""
            if kw_col:
                v = row.get(kw_col, "")
                keyword_val = "" if pd.isna(v) else str(v).strip()
            location_val = ""
            if loc_col:
                v = row.get(loc_col, "")
                location_val = "" if pd.isna(v) else str(v).strip()

            base_out: Dict[str, Any] = {"row_index": i}
            for c in df_work.columns:
                val = row.get(c)
                base_out[f"input_{c}"] = "" if pd.isna(val) else val

            progress.progress((i + 1) / max(n_total, 1), text=f"Ligne {i + 1} / {n_total}")

            if len(text_val) < 5:
                base_out["is_disaster"] = None
                base_out["score"] = None
                base_out["clean_text"] = ""
                base_out["model_name"] = ""
                base_out["api_error"] = "Texte absent ou trop court (minimum 5 caracteres)."
                results.append(base_out)
                continue

            payload = {"text": text_val, "keyword": keyword_val, "location": location_val}
            try:
                status_box.info(f"Prediction en cours — ligne {i + 1}/{n_total}...")
                pred = call_api(payload)
                base_out["is_disaster"] = pred.get("is_disaster")
                base_out["score"] = pred.get("confidence", pred.get("score"))
                base_out["clean_text"] = pred.get("clean_text", "")
                base_out["model_name"] = pred.get("model_name", "")
                base_out["api_error"] = ""
            except Exception as exc:
                base_out["is_disaster"] = None
                base_out["score"] = None
                base_out["clean_text"] = ""
                base_out["model_name"] = ""
                base_out["api_error"] = str(exc)

            results.append(base_out)

        progress.progress(1.0, text="Termine")
        status_box.success(f"Analyse terminee — {len(results)} ligne(s).")
        out_df = pd.DataFrame(results)
        st.session_state["batch_results_df"] = out_df

    if "batch_results_df" in st.session_state and st.session_state["batch_results_df"] is not None:
        out_df = st.session_state["batch_results_df"]
        st.subheader("Resultats")
        st.dataframe(out_df, width="stretch", hide_index=True)
        csv_bytes = out_df.to_csv(index=False).encode("utf-8-sig")
        actions_left, actions_right = st.columns([1.15, 1])
        with actions_left:
            st.download_button(
                label="Telecharger les resultats (CSV)",
                data=csv_bytes,
                file_name="predictions_batch_results.csv",
                mime="text/csv",
                type="primary",
                key="batch_download_csv",
            )
        with actions_right:
            if st.button("Visualiser en bloc", width="stretch", key="batch_view_block_button"):
                st.session_state["batch_show_block_view"] = not st.session_state.get("batch_show_block_view", False)

        if st.session_state.get("batch_show_block_view", False):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            ok_mask = out_df.get("api_error", "").astype(str).str.strip().eq("")
            success_df = out_df[ok_mask] if "api_error" in out_df.columns else out_df
            total_rows = len(out_df)
            success_rows = len(success_df)
            disaster_rows = 0
            if "is_disaster" in success_df.columns:
                disaster_rows = int(success_df["is_disaster"].fillna(False).astype(bool).sum())
            non_disaster_rows = max(0, success_rows - disaster_rows)
            error_rows = max(0, total_rows - success_rows)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Lignes totales", str(total_rows))
            m2.metric("Catastrophe (Oui)", str(disaster_rows))
            m3.metric("Catastrophe (Non)", str(non_disaster_rows))
            m4.metric("Erreurs", str(error_rows))

            summary_lines: List[str] = []
            for _, row in out_df.iterrows():
                idx = row.get("row_index", "")
                txt = str(row.get("input_text", "")).strip()
                txt = txt.replace("\n", " ")
                if len(txt) > 120:
                    txt = txt[:117] + "..."
                if str(row.get("api_error", "")).strip():
                    status = "ERREUR"
                    score_txt = "-"
                else:
                    status = "OUI" if bool(row.get("is_disaster", False)) else "NON"
                    score_raw = row.get("score", "")
                    try:
                        score_txt = f"{float(score_raw):.4f}"
                    except Exception:
                        score_txt = str(score_raw)
                summary_lines.append(f"Ligne {idx} | Catastrophe: {status} | Score: {score_txt} | Texte: {txt}")

            st.markdown("**Vue en bloc (résumé ligne par ligne)**")
            st.text_area(
                "Résumé bloc",
                value="\n".join(summary_lines),
                height=320,
                key="batch_block_summary_textarea",
                label_visibility="collapsed",
            )
            st.markdown("</div>", unsafe_allow_html=True)


def sidebar_controls() -> Dict[str, Any]:
    """Colonne de gauche : logo, déconnexion, URL de l'API, limite de lignes, test /health."""
    logo_path = ROOT_DIR / "image" / "logo.png"
    if logo_path.exists():
        _, center, _ = st.sidebar.columns([1, 2, 1])
        with center:
            st.image(str(logo_path), width="stretch")
        st.sidebar.write("")
    if st.sidebar.button("🔓 Deconnexion", width="stretch", key="logout_button"):
        st.session_state["is_authenticated"] = False
        st.rerun()
    section_title("icon-config", "Configuration", sidebar=True)
    api_url = st.sidebar.text_input("URL API", value=API_URL, key="sidebar_api_url")
    max_batch_rows = st.sidebar.number_input(
        "Lignes max (analyse par lot CSV)",
        min_value=10,
        max_value=5000,
        value=500,
        step=50,
        key="sidebar_max_batch",
    )
    section_title("icon-health", "Etat systeme", sidebar=True)
    health_url = health_url_from_predict_url(api_url)
    if st.sidebar.button(
        "Réveiller l'API maintenant",
        width="stretch",
        key="sidebar_warmup_api",
        help="Force plusieurs appels à /health pour réveiller le service (ex. Render en veille). Peut prendre 1 à 2 minutes.",
    ):
        warm_box = st.sidebar.empty()
        warm_box.info("Réveil en cours… Le premier démarrage du serveur peut prendre 1 à 2 minutes.")
        last_err = ""
        ok = False
        model_loaded_msg = ""
        for i in range(12):
            try:
                r = requests.get(health_url, timeout=60)
                if r.ok:
                    ok = True
                    try:
                        data = r.json()
                        ml = data.get("model_loaded")
                        hf = data.get("heuristic_fallback_active")
                        bits = []
                        if ml is not None:
                            bits.append(f"MLflow chargé : {'oui' if ml else 'non'}")
                        if hf:
                            bits.append("repli heuristique actif")
                        if data.get("model_load_error") and not ml:
                            bits.append("voir logs modèle (model_load_error)")
                        model_loaded_msg = (" " + " · ".join(bits)) if bits else ""
                    except Exception:
                        model_loaded_msg = ""
                    break
                last_err = f"HTTP {r.status_code}"
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
            if i < 11:
                time.sleep(5)
        if ok:
            warm_box.success(f"API joignable.{model_loaded_msg}")
        else:
            warm_box.warning(
                f"Le service ne répond pas encore comme prévu. Réessayez dans un instant ou relancez l'analyse. ({last_err})"
            )
    try:
        health = requests.get(health_url, timeout=12)
        if health.ok:
            st.sidebar.success("API connectee")
        else:
            st.sidebar.warning("API repond avec un statut inattendu")
    except Exception as exc:
        st.sidebar.error(f"API non joignable ({type(exc).__name__})")
    return {"api_url": api_url, "max_batch_rows": int(max_batch_rows)}


def main() -> None:
    """Point d'entrée : mise en page → connexion → sinon barre latérale + deux onglets d'analyse."""
    setup_page()
    _init_auth_state()
    if not st.session_state.get("is_authenticated", False):
        _render_login_screen()
        return

    controls = sidebar_controls()
    global API_URL
    API_URL = controls["api_url"]
    render_header()
    tab_manual, tab_batch = st.tabs(["📝 Analyse manuelle", "📊 Analyse par lot (CSV)"])
    with tab_manual:
        manual_prediction()
    with tab_batch:
        batch_csv_prediction(controls["max_batch_rows"])


if __name__ == "__main__":
    main()

