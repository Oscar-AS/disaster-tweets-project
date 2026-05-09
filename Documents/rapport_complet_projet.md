# 📊 Rapport de Projet Détaillé : Classification de Tweets de Catastrophes

Ce rapport est conçu pour offrir une compréhension approfondie du projet, en expliquant non seulement **ce qui a été fait**, mais aussi **pourquoi** ces choix technologiques ont été faits.

---

## 1. Contexte et Enjeux
Le projet s'inscrit dans le domaine du **Natural Language Processing (NLP)**. L'enjeu est de traiter le flux massif de Twitter pour identifier instantanément les signaux de crises réelles.

*   **Le défi :** Twitter est riche en métaphores. Un tweet disant *"This party is fire!"* ne doit pas être classé comme une catastrophe, contrairement à *"The forest is on fire!"*. Le modèle doit donc comprendre le **contexte**.

---

## 2. Préparation des Données (Le Pipeline de Nettoyage)
La donnée brute sur Twitter est "bruitée". Nous avons mis en place un pipeline de nettoyage rigoureux :
- **Normalisation :** Conversion en minuscules pour que "Feu" et "feu" soient traités de la même manière.
- **Suppression du bruit :** Retrait des liens (http...), des pseudos (@user) et des caractères spéciaux qui n'apportent pas de sens sémantique.
- **Gestion des Stop-words :** Retrait des mots très fréquents (le, la, et...) pour se concentrer sur les mots porteurs de sens (incendie, aide, blessés).
- **Tokenisation :** Découpage des phrases en unités (mots ou sous-mots) que la machine peut traiter.

---

## 3. L'Évolution des Modèles (Du plus simple au plus complexe)

### A. L'Approche Statistique (TF-IDF + Sklearn)
Nous avons commencé par transformer le texte en chiffres via la méthode **TF-IDF**. 
*   **Principe :** Plus un mot est rare et présent dans un tweet, plus il a d'importance.
*   **Limites :** Cette méthode ne comprend pas l'ordre des mots ni le contexte.

### B. L'Approche Séquentielle (Deep Learning - LSTM/CNN)
Pour pallier les limites de TF-IDF, nous avons utilisé des **Embeddings** (vecteurs mathématiques représentant le sens des mots) :
*   **LSTM (Long Short-Term Memory) :** Un réseau de neurones avec une "mémoire" capable de comprendre la structure d'une phrase complète.
*   **TextCNN :** Utilise des filtres (comme pour les images) pour détecter des expressions clés ou des combinaisons de mots alarmantes.

### C. La Révolution des Transformers (BERT, BERTweet)
C'est l'approche la plus moderne utilisant le mécanisme d'**Attention**.
*   **Principe :** Le modèle regarde chaque mot par rapport à tous les autres mots de la phrase simultanément pour en déduire le sens exact.
*   **BERTweet :** C'est notre champion. Contrairement au BERT classique (entraîné sur Wikipedia), BERTweet a été entraîné sur 850 millions de tweets. Il comprend donc l'argot, les abréviations et le style spécifique de Twitter.

---

## 4. La Stratégie d'Optimisation : Le F2-Score
Dans ce projet, l'erreur n'a pas le même coût selon le sens :
1.  **Faux Positif :** Classer un tweet banal comme "catastrophe". (Conséquence : Un peu de temps perdu pour les secours).
2.  **Faux Négatif :** Rater un tweet de catastrophe réelle. (**Conséquence : Des vies en danger**).

C'est pourquoi nous avons optimisé le **F2-Score** : une métrique mathématique qui pénalise beaucoup plus lourdement les oublis (Faux Négatifs) que les fausses alertes.

---

## 5. Architecture Technique (MLOps)
Pour transformer ce modèle en un outil réel, nous avons construit une infrastructure complète :

*   **DagsHub (Le Centre de Contrôle) :** C'est le "GitHub pour le Machine Learning". Il héberge notre serveur **MLflow** et nous permet de centraliser tous les modèles entraînés par l'équipe. Cela permet à tout le monde de collaborer sur les mêmes données et résultats.
*   **FastAPI (Le Moteur Intelligent) :** 
    *   **Chargement Dynamique :** Contrairement aux APIs classiques, celle-ci n'a pas le modèle stocké "en dur" dans son code. 
    *   **Lien Direct avec le Cloud :** Elle se connecte directement à **DagsHub/MLflow** au démarrage pour télécharger automatiquement la version marquée comme **"Champion"** (Production).
    *   **Avantage :** Si l'équipe entraîne un meilleur modèle, il suffit de changer son étiquette sur DagsHub, et l'API se mettra à jour automatiquement au prochain redémarrage, sans avoir à modifier une seule ligne de code.
*   **Streamlit (L'Interface de Visualisation) :** Une application web interactive où l'utilisateur peut taper un tweet et voir instantanément si le modèle le juge dangereux, avec des graphiques de confiance.
*   **Docker & Render (Le Déploiement) :** Permet d'emballer toute l'application dans une "boîte" (conteneur) pour qu'elle soit accessible en ligne 24h/24, partout dans le monde.

---

## 6. Conclusion pour la Présentation
Le projet démontre qu'en combinant des modèles de pointe (**BERTweet**) avec une stratégie centrée sur la sécurité (**F2-Score**) et une infrastructure moderne, on peut transformer le bruit des réseaux sociaux en un outil d'aide à la décision fiable pour les services d'urgence.
