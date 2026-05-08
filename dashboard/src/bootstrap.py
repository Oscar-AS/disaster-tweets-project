"""
================================================================================
Démarrage automatique des bibliothèques Python (pour lecteur non technique)
================================================================================
Ce petit fichier sert avant tout à éviter une erreur du type « module introuvable »
quand on ouvre l'application sur un ordinateur où certains paquets ne sont pas
encore installés. Le programme vérifie si les bibliothèques nécessaires existent ;
sinon, il lance une installation en arrière-plan (commande `pip install`).

Ce n'est pas la méthode utilisée en production « sérieuse » (on préfère un fichier
requirements figé), mais c'est pratique pour faire tourner le tableau de bord
rapidement après un clone du projet.
================================================================================
"""

# Vérifie si un module est présent sans l'importer complètement
import importlib.util
# Permet de lancer la commande d'installation des paquets comme le ferait un humain dans un terminal
import subprocess
# Accès à l'interpréteur Python actuel (pour appeler `python -m pip ...`)
import sys
import warnings
from typing import Iterable, Tuple


def _is_installed(import_name: str) -> bool:
    """
    Pour un non-technicien : répond « oui » si la bibliothèque `import_name`
    est déjà disponible sur cette machine, « non » sinon.
    """
    return importlib.util.find_spec(import_name) is not None


def ensure_packages(packages: Iterable[Tuple[str, str]]) -> None:
    """
    Pour un non-technicien : reçoit une liste de paires (nom pour pip, nom pour import).
    Exemple : ("pandas", "pandas") signifie « si import pandas échoue, exécute pip install pandas ».
    Si tout est déjà installé, cette fonction ne fait rien.
    """
    missing = [pip_name for pip_name, import_name in packages if not _is_installed(import_name)]
    if not missing:
        return

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    except subprocess.CalledProcessError as exc:
        # Sur certains environnements (ex: Streamlit Cloud), installer à chaud peut échouer.
        # On évite de bloquer l'app au démarrage; la vraie source de vérité reste requirements.txt.
        warnings.warn(
            "Installation automatique des dépendances impossible. "
            f"Paquets manquants: {missing}. Détail: {exc}. "
            "Ajoutez-les dans requirements.txt puis redéployez.",
            RuntimeWarning,
            stacklevel=2,
        )
