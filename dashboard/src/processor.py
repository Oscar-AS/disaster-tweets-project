# "re" sert a nettoyer le texte avec des expressions regulieres.
import re
# Types de retour attendus pour rendre le code plus lisible.
from typing import List, Tuple

# spaCy est utilise pour detecter des entites nommees (villes, lieux).
import spacy


# Petit referentiel interne: nom de ville -> coordonnees [latitude, longitude].
_KNOWN_LOCATIONS = {
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


# Classe qui gere le pretraitement NLP (nettoyage + detection de lieux).
class TextProcessor:
    """Cleaning and lightweight geolocation extraction with spaCy NER."""

    # Constructeur: charge le modele spaCy, sinon cree une version minimale.
    def __init__(self) -> None:
        try:
            # Modele anglais standard avec NER deja entraine.
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback si le modele n'est pas installe.
            self.nlp = spacy.blank("en")
            # On ajoute le composant NER si absent.
            if "ner" not in self.nlp.pipe_names:
                self.nlp.add_pipe("ner")

    # Methode utilitaire statique: nettoie le texte brut.
    @staticmethod
    def clean_text(text: str) -> str:
        # Retire espaces inutiles debut/fin.
        text = text.strip()
        # Supprime les liens web.
        text = re.sub(r"http\S+|www\S+", "", text)
        # Supprime les mentions Twitter (@utilisateur).
        text = re.sub(r"@\w+", "", text)
        # Remplace les multiples espaces par un seul.
        text = re.sub(r"\s+", " ", text)
        # Retourne le texte propre.
        return text

    # Extrait les lieux sous forme: [("Nom", [lat, lon]), ...]
    def extract_locations(self, text: str) -> List[Tuple[str, List[float]]]:
        # Analyse linguistique du texte.
        doc = self.nlp(text)
        # Liste temporaire des lieux detectes.
        locations = []

        # Parcourt toutes les entites detectees par spaCy.
        for ent in doc.ents:
            # On garde les entites de type lieu (ville, zone geo, infrastructure).
            if ent.label_ in {"GPE", "LOC", "FAC"}:
                # Cle normalisee pour comparer avec notre dictionnaire.
                key = ent.text.lower().strip()
                # Si on connait cette ville, on ajoute ses coordonnees.
                if key in _KNOWN_LOCATIONS:
                    locations.append((ent.text, _KNOWN_LOCATIONS[key]))

        # Si spaCy n'a rien trouve, on fait une recherche simple par mots connus.
        if not locations:
            for city, coords in _KNOWN_LOCATIONS.items():
                if city in text.lower():
                    locations.append((city.title(), coords))

        # On supprime les doublons avec une cle en minuscule.
        dedup = {}
        for place, coords in locations:
            dedup[place.lower()] = (place, coords)
        # On renvoie la liste finale sans doublons.
        return list(dedup.values())

