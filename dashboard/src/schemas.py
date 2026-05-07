# Dict et List servent a typer les donnees de sortie.
from typing import Dict, List

# BaseModel cree des objets valides automatiquement; Field fixe des regles; field_validator ajoute une verification personnalisee.
from pydantic import BaseModel, Field, field_validator


# Schema des donnees recues par l'API.
class PredictRequest(BaseModel):
    # Le champ "text" est obligatoire (...), avec longueur minimale 5 et maximale 280 caracteres.
    text: str = Field(..., min_length=5, max_length=280)
    # Mot-cle optionnel associe au tweet.
    keyword: str = Field(default="", max_length=80)
    # Emplacement optionnel associe au tweet.
    location: str = Field(default="", max_length=120)

    # Cette validation supplementaire supprime les espaces au debut/fin.
    @field_validator("text")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        # On retire les espaces inutiles.
        cleaned = value.strip()
        # Si apres nettoyage le texte est vide, on bloque la requete.
        if not cleaned:
            raise ValueError("Le texte ne doit pas etre vide.")
        # On renvoie la version nettoyee.
        return cleaned

    @field_validator("keyword", "location")
    @classmethod
    def normalize_optional_fields(cls, value: str) -> str:
        return value.strip()


# Schema des donnees renvoyees par l'API.
class PredictResponse(BaseModel):
    # True si le tweet est estime comme catastrophe.
    is_disaster: bool
    # Score de confiance entre 0 et 1.
    score: float
    # Dictionnaire mot -> contribution au score.
    impact_words: Dict[str, float]
    # Liste des coordonnees detectees: [[lat, lon], [lat, lon], ...]
    geo_coords: List[List[float]]

