# Application de detection de tweets de catastrophe

Ce projet contient une application web simple qui aide a analyser des tweets pour estimer s'ils parlent d'une catastrophe (incendie, inondation, explosion, etc.).

## A quoi sert l'application ? (version non technique)

L'application sert a :
- saisir un tweet manuellement et obtenir un resultat de risque,
- envoyer un fichier CSV de plusieurs tweets pour une analyse en lot,
- afficher un score, un resultat Oui/Non, et des visualisations (graphe + carte),
- telecharger les resultats en CSV.

En clair : vous donnez du texte, l'application vous aide a prioriser les messages potentiellement urgents.

## Comment ca fonctionne, simplement

1. Vous ouvrez l'application Streamlit dans le navigateur.
2. Vous vous connectez avec un mot de passe admin.
3. Vous choisissez :
   - **Analyse manuelle** (un tweet),
   - **Analyse par lot CSV** (plusieurs tweets).
4. Le texte est envoye a une API de prediction.
5. L'API renvoie :
   - `is_disaster` : oui/non,
   - `confidence` ou `score` : niveau de confiance,
   - parfois des infos complementaires (texte nettoye, nom du modele, etc.).
6. L'interface affiche le resultat et vous permet d'exporter les donnees.

## Traduction automatique

Si vous saisissez un texte non anglais :
- l'application detecte la langue,
- traduit le texte en anglais,
- affiche la traduction avant prediction,
- puis envoie cette version traduite au modele.

## Fonctions principales de l'interface

- **Connexion securisee** par mot de passe.
- **Etat API** dans la barre laterale (connectee ou non).
- **Bouton "Reveiller l'API maintenant"** pour les services qui se mettent en veille.
- **Analyse manuelle** avec affichage detaille du resultat.
- **Analyse CSV** avec :
  - detection flexible des colonnes texte (`text`, `tweet`, `message`, `content`, `texte`),
  - bouton de lancement,
  - tableau de resultats,
  - bouton de telechargement CSV,
  - vue "Visualiser en bloc".

## Structure du projet (simplifiee)

```text
ML/
├── app/
│   └── streamlit_app.py        # application web
├── src/
│   ├── bootstrap.py            # aide dependances au demarrage
│   └── model.py                # logique modele locale (selon usage)
├── disaster-tweets-project/
│   └── API/main.py             # API FastAPI de prediction
├── docker/
│   └── start_services.sh
├── Dockerfile
├── requirements.txt
└── README.md
```

## Installation rapide (local)

```bash
python -m venv .venv
```

Activez l'environnement :

- Windows PowerShell :
```powershell
.venv\Scripts\Activate.ps1
```

- Linux / Mac :
```bash
source .venv/bin/activate
```

Installez les dependances :

```bash
pip install -r requirements.txt
```

Lancez l'application :

```bash
streamlit run app/streamlit_app.py
```

## URL utiles

- Application : `http://localhost:8501`
- API distante actuelle (par defaut) : `https://disaster-tweets-project.onrender.com/predict`

## Limitations a connaitre

- Le score est une estimation, pas une verite absolue.
- Si le service API est en veille, la premiere requete peut prendre du temps.
- Si le modele distant n'est pas charge, la prediction peut echouer temporairement.

## Depannage rapide

- **Erreur 503** : utilisez "Reveiller l'API maintenant", puis reessayez.
- **CSV refuse** : verifiez les noms de colonnes attendus.
- **Erreur de dependance** : reexecutez `pip install -r requirements.txt`.

