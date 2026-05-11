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

## Detail des 2 packages de traduction

Cette section explique les 2 bibliotheques utilisees dans ce projet :
- `langdetect` (detecter la langue),
- `deep-translator` (traduire le texte).

### 1) Package `langdetect`

**Role simple**
- Deviner automatiquement la langue d'un texte (fr, en, es, etc.).

**Fonctions utilisees dans le projet**
- `detect(text: str) -> str`
  - Retourne le code langue le plus probable (`"fr"`, `"en"`, ...).
  - Utilisee avant la traduction pour savoir s'il faut traduire.
- `LangDetectException`
  - Exception levee quand le texte est trop court, vide ou ambigu.
  - Capturee pour eviter de bloquer l'application.

**Fonctions importantes (API du package)**
- `detect_langs(text: str) -> list`
  - Retourne plusieurs langues avec probabilites.
  - Exemple de sortie : `[fr:0.92, en:0.08]`.

### 2) Package `deep-translator`

**Role simple**
- Traduire du texte d'une langue source vers une langue cible.

**Classe/fonctions utilisees dans le projet**
- `GoogleTranslator(source=..., target=...)`
  - Cree un traducteur (ex: source `fr`, cible `en`).
- `GoogleTranslator(...).translate(text: str) -> str`
  - Traduit un seul texte.
  - C'est la fonction centrale utilisee pour la prediction.

**Fonctions utiles du package**
- `GoogleTranslator(...).translate_batch(list_of_texts: list[str]) -> list[str]`
  - Traduit plusieurs textes d'un coup.
- `GoogleTranslator.get_supported_languages(as_dict=False)`
  - Retourne les langues supportees.

### Mecanisme complet (etape par etape)

1. L'utilisateur saisit un texte (par ex. francais).
2. L'app verifie si le texte est assez long pour une detection fiable (`TRANSLATE_MIN_CHARS`).
3. `langdetect.detect(text)` identifie la langue.
4. Si la langue est deja anglaise (`en`), on garde le texte original.
5. Sinon, `deep-translator` traduit vers l'anglais.
6. L'app affiche la traduction (transparence utilisateur).
7. La prediction est faite sur le texte traduit.
8. Le resultat (Oui/Non + score) est affiche dans le dashboard.

### Gestion des erreurs (important)

- Si la detection echoue (`LangDetectException`) : on garde le texte original.
- Si la traduction echoue (reseau/API externe) : on garde le texte original.
- Si l'utilisateur desactive la traduction (`DISABLE_AUTO_TRANSLATION=true`) : aucune traduction n'est tentee.

### Variables de configuration liees a la traduction

- `DISABLE_AUTO_TRANSLATION`
  - `true/1/yes` : desactive la traduction automatique.
- `TRANSLATE_MIN_CHARS`
  - longueur minimale du texte pour tenter detection + traduction (ex: `8`).

### Pourquoi ce design est adapte a un non technicien

- On evite les blocages : en cas d'echec traduction, l'app continue.
- On garde la transparence : la phrase traduite est visible avant prediction.
- On garde la simplicite : 2 outils specialises, chacun pour une seule tache.

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

