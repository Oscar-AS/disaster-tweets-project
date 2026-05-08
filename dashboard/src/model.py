# "os" permet de lire des chemins de fichiers et variables d'environnement.
import os
# OrderedDict aide a conserver l'ordre d'ajout des impacts de mots.
from collections import OrderedDict
# Types Python pour clarifier les entrees/sorties.
from typing import Any, Dict, Tuple
try:
    import numpy as np
except ModuleNotFoundError:  # fallback: MLflow non utilise / pas installe
    np = None  # type: ignore
try:
    import pandas as pd
except ModuleNotFoundError:  # fallback: MLflow non utilise / pas installe
    pd = None  # type: ignore

# joblib charge le modele ML deja entraine et sauvegarde en .joblib.
import joblib

# mlflow est utilise pour charger le meilleur modele depuis le MLflow Registry.
try:
    import mlflow
    import mlflow.pyfunc
except ModuleNotFoundError:  # fallback: MLflow non utilise / pas installe
    mlflow = None  # type: ignore


# Classe principale de prediction de catastrophe.
class DisasterDetector:
    """Inference with a pre-trained .joblib model and token attribution."""

    # Constructeur: prepare le chemin du modele et charge le modele.
    def __init__(self, model_path: str | None = None) -> None:
        # Priorite: parametre passe -> variable d'env MODEL_PATH -> chemin par defaut.
        self.model_path = model_path or os.getenv("MODEL_PATH", "models/disaster_model.joblib")
        # Modele non charge au depart.
        self.model: Any | None = None
        # Modele MLflow pyfunc non charge au depart.
        self.mlflow_model: Any | None = None

        # Tente d'abord de charger le modele joblib local (rapide).
        self._load_model()
        # Si joblib absent ou non charge, tente de charger depuis MLflow (fallback).
        if self.model is None:
            self._load_mlflow_model()
        # Liste de mots associes aux catastrophes (utile pour fallback).
        self.disaster_terms = {
            "earthquake",
            "flood",
            "wildfire",
            "fire",
            "hurricane",
            "evacuation",
            "disaster",
            "collapsed",
            "injured",
            "dead",
            "tsunami",
            "explosion",
            "rescue",
            "storm",
        }

    def _load_mlflow_model(self) -> None:
        """
        Tente de charger le modele depuis le MLflow Model Registry (DagsHub).
        Fallback grace au try/except: si echec, on continue avec joblib/rules.
        """
        if os.getenv("DISABLE_MLFLOW", "").lower() in {"1", "true", "yes"}:
            return
        if mlflow is None:
            return
        if pd is None or np is None:
            return

        model_name = os.getenv("MLFLOW_MODEL_NAME", "Disaster_Tweet_Predictor_Prod")
        stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")
        model_uri = os.getenv("MLFLOW_MODEL_URI") or f"models:/{model_name}/{stage}"

        # Tracking URI: si pas defini, on utilise dagshub pour initialiser proprement.
        # (dagshub peut etre optionnel si MLFLOW_TRACKING_URI est fourni).
        try:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            else:
                try:
                    import dagshub  # import local pour ne pas penaliser si non necessaire

                    dagshub.init(repo_owner="Oscar-AS", repo_name="disaster-tweets-project", mlflow=True)
                except Exception:
                    # Sans tracking_uri explicite, mlflow tente ses valeurs par defaut.
                    pass

            self.mlflow_model = mlflow.pyfunc.load_model(model_uri)
        except Exception:
            # Si le chargement MLflow echoue, on revient au fallback regles/joblib.
            self.mlflow_model = None

    # Charge le fichier modele s'il existe.
    def _load_model(self) -> None:
        # Verifie que le fichier est present.
        if os.path.exists(self.model_path):
            # Charge l'objet Python (pipeline ou estimateur) depuis le .joblib.
            self.model = joblib.load(self.model_path)

    # Calcule un score de probabilite de catastrophe entre 0 et 1.
    def _base_score(self, text: str) -> float:
        # Cas MLflow: on convertit la sortie du pyfunc en probabilite [0..1].
        if self.mlflow_model is not None:
            return self._mlflow_predict_score(text)

        # Expected model interface:
        # - scikit-learn pipeline with predict_proba, OR
        # - estimator with decision_function / predict
        # Si un modele est charge, on l'utilise en priorite.
        if self.model is not None:
            # Cas ideal: le modele renvoie directement des probabilites.
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba([text])[0]
                # On prend la probabilite de la classe positive (catastrophe).
                return float(proba[-1])
            # Sinon, on convertit un score brut (decision_function) en probabilite.
            if hasattr(self.model, "decision_function"):
                raw = float(self.model.decision_function([text])[0])
                return 1.0 / (1.0 + pow(2.718281828, -raw))
            # Dernier recours: prediction binaire 0/1.
            if hasattr(self.model, "predict"):
                pred = int(self.model.predict([text])[0])
                return float(pred)

        # Fallback: lightweight rule-based estimate if model file is absent.
        # Si aucun modele n'est charge, on applique une regle simple basee sur des mots-cles.
        words = [w.strip(".,!?").lower() for w in text.split()]
        # Chaque mot de catastrophe ajoute un bonus.
        bonus = sum(1 for w in words if w in self.disaster_terms) * 0.08
        # On borne le score final entre 0 et 1.
        return max(0.0, min(1.0, 0.2 + bonus))

    def _mlflow_predict_score(self, text: str) -> float:
        """
        Lance la prediction MLflow et normalise le resultat vers une probabilite
        pour la classe positive (catastrophe).
        """
        if self.mlflow_model is None or pd is None or np is None:
            raise RuntimeError("MLflow non disponible (mlflow_model/numpy/pandas manquants).")

        # Formats d'entree courants pour MLflow pyfunc (selon le modele logge).
        input_candidates = [pd.Series([text]), [text], np.array([text])]

        last_exc: Exception | None = None
        for input_data in input_candidates:
            try:
                prediction = self.mlflow_model.predict(input_data)
                score = self._extract_disaster_score_from_prediction(prediction)
                if score is not None:
                    return score
            except Exception as exc:
                last_exc = exc

        # Si rien ne marche, on remonte l'erreur principale (ou un message generic).
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Erreur lors de la prediction MLflow (sortie non interpretable).")

    def _extract_disaster_score_from_prediction(self, prediction: Any) -> float | None:
        """
        Convertion defensive du resultat pyfunc MLflow vers un score scalaire [0..1].
        """
        if pd is None or np is None:
            return None
        # 1) HuggingFace / classifiers encapsules: DataFrame avec 'label'/'score'
        if isinstance(prediction, pd.DataFrame):
            if "score" not in prediction.columns:
                return None

            # Si 'label' existe, on essaie de cibler la classe positive.
            if "label" in prediction.columns and len(prediction) >= 1:
                positive_labels = {"LABEL_1", "1", "DISASTER", "DISASTERS", "TRUE", "CAT", "DIS_CAT", "YES"}
                negative_labels = {"LABEL_0", "0", "NON_DISASTER", "FALSE", "NOT_DISASTER", "NO"}

                # Cherche explicitement une ligne positive.
                for _, row in prediction.iterrows():
                    label = str(row["label"]).upper()
                    if label in positive_labels:
                        return max(0.0, min(1.0, float(row["score"])))

                # Cherche explicitement une ligne negative, si elle existe.
                for _, row in prediction.iterrows():
                    label = str(row["label"]).upper()
                    if label in negative_labels:
                        return max(0.0, min(1.0, 1.0 - float(row["score"])))

                # Fallback: on prend le max score (souvent correspond a la classe la plus probable).
                scores = [float(s) for s in prediction["score"].tolist()]
                if scores:
                    return max(0.0, min(1.0, max(scores)))

            # Si pas de 'label', on suppose que le/les score(s) renvoyes correspondent au positif.
            row0 = prediction.iloc[0]
            return max(0.0, min(1.0, float(row0["score"])))

        # 2) Liste de dictionnaires: [{'label': ..., 'score': ...}]
        if isinstance(prediction, list) and prediction and isinstance(prediction[0], dict):
            res = prediction[0]
            if "score" in res:
                score = float(res["score"])
                label = str(res.get("label", "")).upper()
                positive_labels = {"LABEL_1", "1", "DISASTER", "DISASTERS", "TRUE", "YES"}
                negative_labels = {"LABEL_0", "0", "NON_DISASTER", "FALSE", "NOT_DISASTER", "NO"}
                if label in positive_labels:
                    return max(0.0, min(1.0, score))
                if label in negative_labels:
                    return max(0.0, min(1.0, 1.0 - score))
                return max(0.0, min(1.0, score))
            return None

        # 3) Keras / Tensor: numpy array de probabilite ou logits
        if isinstance(prediction, np.ndarray):
            arr = prediction.astype(float).flatten()
            if arr.size == 0:
                return None
            # Cas [p0, p1]
            if arr.size >= 2:
                # Si c'est manifestement deja des probabilites, on prend la derniere.
                if np.all(arr >= 0.0) and np.all(arr <= 1.0):
                    return float(max(0.0, min(1.0, arr[-1])))
                # Sinon, on prend un maximum par prudence (souvent correspond a la classe positive en sortie brute).
                return float(max(0.0, min(1.0, arr.max())))
            return float(max(0.0, min(1.0, arr[0])))

        # 4) Autres types: on tente de convertir en array puis extraire un scalaire
        try:
            arr = np.array(prediction, dtype=float).flatten()
            if arr.size == 0:
                return None
            return float(max(0.0, min(1.0, arr[0])))
        except Exception:
            return None

    # Donne une explication simple: impact estime de chaque mot sur le score.
    def _impact_words(self, text: str, reference_score: float) -> Dict[str, float]:
        # MLflow peut etre beaucoup plus lent; l'ablation mot-par-mot peut devenir trop couteuse.
        if self.mlflow_model is not None:
            return self._impact_words_heuristic(text, reference_score)

        # Decoupe le texte en mots.
        words = [w for w in text.split() if w]
        # Dictionnaire ordonne des impacts.
        impacts = OrderedDict()
        # Limite pour garder des temps de calcul raisonnables.
        max_words = min(12, len(words))

        # Pour chaque mot, on retire ce mot puis on mesure la baisse/hausse du score.
        for i in range(max_words):
            word = words[i]
            ablated = " ".join(words[:i] + words[i + 1 :]).strip() or text
            score_without = self._base_score(ablated)
            impact = round(reference_score - score_without, 4)
            impacts[word] = impact

        # Trie par importance absolue pour afficher d'abord les mots les plus influents.
        return dict(sorted(impacts.items(), key=lambda item: abs(item[1]), reverse=True))

    def _impact_words_heuristic(self, text: str, reference_score: float) -> Dict[str, float]:
        """
        Attribution rapide quand on utilise un modele MLflow (evite plusieurs appels reseau).
        Ici on fournit une contribution approximative basee sur la presence de termes de catastrophe.
        """
        tokens = [w.strip(".,!?").lower() for w in text.split() if w]
        impacts: Dict[str, float] = {}

        for tok in tokens[:12]:
            if tok in self.disaster_terms:
                # On scale legerement autour du reference_score pour garder un ordre plus ou moins cohérent.
                impacts[tok] = round(max(0.01, min(0.3, 0.12 * reference_score)), 4)

        # On trie par valeur decroissante pour bien remplir le graphique.
        return dict(sorted(impacts.items(), key=lambda item: abs(item[1]), reverse=True))

    # Fonction appelee par l'API: renvoie decision finale + score + explications.
    def predict(self, text: str) -> Tuple[bool, float, Dict[str, float]]:
        # Score global.
        score = round(self._base_score(text), 4)
        # Contributions des mots.
        impact_words = self._impact_words(text, score)
        # Regle de decision: catastrophe si score >= 0.5.
        is_disaster = score >= 0.5
        # Retour du resultat complet.
        return is_disaster, score, impact_words

