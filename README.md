
# Disaster Tweets Project - Guide de démarrage

Ce dépôt contient notre projet de classification de tweets de catastrophes. Merci de suivre scrupuleusement ces étapes pour maintenir un code propre.

---

## . Installation initiale (À faire une seule fois)

```bash
# Cloner le projet
git clone https://github.com/Oscar-AS/disaster-tweets-project.git
cd disaster-tweets-project

# Installer les outils
pip install -r requirements.txt
```
---

## 2. Gestion des Branches

Avant de coder, vérifie toujours où tu es et synchronise-toi avec le serveur.

```bash
# Voir toutes les branches (locales et sur GitHub)
git branch -a

# Mettre à jour la liste des branches si tu ne vois pas les nouvelles
git fetch --all

# Migrer vers une branche existante
git checkout nom_de_la_branche

# OU Créer ta propre branche pour une nouvelle tâche
git checkout -b feat-nom-de-ta-tache
```

---

## 3. Workflow Quotidien (Le cycle de vie du code)

### Début de session : Toujours récupérer le travail des autres
```bash
git checkout main
git pull origin main
git checkout ta-branche
git pull origin ta-branche
```

### 📤 Fin de session : Sauvegarder ton travail
```bash
# 1. Vérifie ta branche pour ne pas push sur main par erreur !
git branch

# 2. Ajoute et valide tes fichiers
git add .
git commit -m "Explique ici ce que tu as fait (ex: ajout du nettoyage de texte)"

# 3. Envoie sur GitHub
git push origin ta-branche
```

---

## Règles d'or (Best Practices)

1. **Vérification de branche :** Tape toujours `git branch` avant un `git add .`.
2. **Pull avant Push :** Assure-toi d'être à jour avant d'envoyer ton code pour éviter les conflits.
3. **Fichiers lourds :** Ne jamais ajouter les fichiers `.csv` de données dans les commits (ils doivent rester dans le dossier `data/` ignoré par Git).
4. **CI/CD :** Si ton push fait apparaître une **croix rouge (❌)** sur GitHub, c'est que ton code a cassé les tests. Corrige-le immédiatement !