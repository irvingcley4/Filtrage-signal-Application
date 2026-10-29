"""Structures de données partagées entre le cœur de traitement et l'interface."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# Un « rappel » est appelé par les traitements : rappel(statut, progression).
# progression est une fraction entre 0 et 1, ou None quand l'avancement réel est
# inconnu (l'interface affiche alors une barre indéterminée).
Rappel = Callable[[str, Optional[float]], None]


def signaler(rappel: Optional[Rappel], statut: str, progression: Optional[float] = None) -> None:
    """Appelle le rappel s'il existe. Il peut lever TraitementAnnule pour interrompre."""
    if rappel is not None:
        rappel(statut, progression)


@dataclass
class Metrique:
    """Une valeur affichée à l'utilisateur.

    nature : « mesure » (calculée contre une référence connue),
             « estime » (approximation sans référence),
             « info » (simple information, pas une note de qualité).
    """
    libelle: str
    valeur: str
    nature: str = "info"
    aide: str = ""


@dataclass
class ResultatTraitement:
    """Ce que renvoie un traitement : fichiers créés, métriques, segments, remarques."""
    dossier: Path
    fichiers: dict[str, Path] = field(default_factory=dict)
    metriques: list[Metrique] = field(default_factory=list)
    segments: list[tuple[float, float, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    duree_calcul_s: float = 0.0
