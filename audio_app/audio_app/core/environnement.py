"""Environnement : FFmpeg, dépendances, dossiers de sortie, ouverture de dossier."""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Optional


def options_subprocess() -> dict:
    """Sous Windows, évite qu'une fenêtre de console clignote à chaque appel de FFmpeg."""
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}


def chemin_ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def chemin_ffprobe() -> Optional[str]:
    return shutil.which("ffprobe")


def ffmpeg_disponible() -> bool:
    return chemin_ffmpeg() is not None


def dependance_disponible(module: str) -> bool:
    """Vrai si le module Python est installable/importable (sans l'importer)."""
    try:
        return importlib.util.find_spec(module) is not None
    except ValueError:                 # module déjà importé mais sans « spec » : il est donc présent
        return module in sys.modules
    except (ImportError, AttributeError):
        return False


def demucs_disponible() -> bool:
    """Vrai si PyTorch ET une version de Demucs (« demucs » ou le fork « demucs_infer ») sont installés."""
    return dependance_disponible("torch") and (
        dependance_disponible("demucs") or dependance_disponible("demucs_infer"))


def message_installation_demucs(detail: str = "") -> str:
    """Message lisible (français) avec les commandes d'installation de Demucs."""
    texte = ("Demucs (séparation par apprentissage profond) n'est pas utilisable : PyTorch ou Demucs "
             "est absent ou ne s'importe pas.\n\n"
             "1) Installez PyTorch (version CPU, voir le README pour le GPU NVIDIA) :\n"
             "   python -m pip install torch\n"
             "2) Installez Demucs, au choix :\n"
             "   python -m pip install demucs-infer      (Python 3.12 / 3.13, PyTorch récent)\n"
             "   python -m pip install demucs            (Python 3.11 ou moins)\n\n"
             "En attendant, REPET-SIM et NMF restent disponibles.")
    return texte + (f"\n\nDétail technique : {detail}" if detail else "")


def commande_installation_ffmpeg() -> str:
    if sys.platform == "win32":
        return "winget install ffmpeg"
    if sys.platform == "darwin":
        return "brew install ffmpeg"
    return "sudo apt install ffmpeg"


@dataclass(frozen=True)
class Dependance:
    nom: str
    role: str
    disponible: bool
    version: str
    commande_installation: str


# (nom affiché, module Python, paquet pip, rôle)
_DEPENDANCES_PYTHON = [
    ("NumPy", "numpy", "numpy", "calcul numérique"),
    ("SciPy", "scipy", "scipy", "filtres, STFT, lecture WAV"),
    ("Matplotlib", "matplotlib", "matplotlib", "spectrogrammes et graphiques"),
    ("scikit-learn", "sklearn", "scikit-learn", "NMF (séparation expérimentale)"),
    ("PySide6", "PySide6", "PySide6", "interface graphique"),
    ("PyTorch", "torch", "torch", "moteur de Demucs (séparation haute qualité)"),
]


def verifier_dependances() -> list[Dependance]:
    """État de chaque dépendance, avec la commande d'installation correspondante."""
    resultat = []
    for nom, module, paquet, role in _DEPENDANCES_PYTHON:
        dispo = dependance_disponible(module)
        try:
            version = metadata.version(paquet) if dispo else ""
        except metadata.PackageNotFoundError:
            version = ""
        resultat.append(Dependance(nom, role, dispo, version, f"python -m pip install {paquet}"))
    demucs_ok = dependance_disponible("demucs") or dependance_disponible("demucs_infer")
    nom_pip = "demucs" if dependance_disponible("demucs") else "demucs-infer"
    try:
        version_demucs = metadata.version(nom_pip) if demucs_ok else ""
    except metadata.PackageNotFoundError:
        version_demucs = ""
    resultat.append(Dependance("Demucs", "séparation voix / instrumental par apprentissage profond",
                               demucs_ok, version_demucs,
                               "python -m pip install demucs-infer  (ou : demucs si Python ≤ 3.11)"))
    ffmpeg = chemin_ffmpeg() is not None
    resultat.append(Dependance("FFmpeg", "lecture MP3/M4A/FLAC/OGG… (pas nécessaire pour les WAV)",
                               ffmpeg, "installé" if ffmpeg else "", commande_installation_ffmpeg()))
    return resultat


def dossier_sortie_par_defaut() -> Path:
    return Path.home() / "AudioSuite_resultats"


def creer_dossier_sortie(base, maintenant: Optional[datetime] = None) -> Path:
    """Crée un sous-dossier horodaté, unique, dans `base` (ex. resultats_2026-10-12_2350).
    Ne réutilise jamais un dossier existant : aucun fichier n'est écrasé."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    horodatage = (maintenant or datetime.now()).strftime("%Y-%m-%d_%H%M")
    numero = 1
    while True:
        suffixe = "" if numero == 1 else f"_{numero}"
        candidat = base / f"resultats_{horodatage}{suffixe}"
        try:
            candidat.mkdir()
            return candidat
        except FileExistsError:
            numero += 1


def ouvrir_dossier(chemin) -> bool:
    """Ouvre un dossier dans l'explorateur du système. Retourne False en cas d'échec.
    Les commandes sont passées en liste (jamais via un shell) : pas d'injection possible."""
    chemin = Path(chemin)
    if not chemin.is_dir():
        return False
    try:
        if sys.platform == "win32":
            os.startfile(str(chemin))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(chemin)], check=False)
        else:
            subprocess.run(["xdg-open", str(chemin)], check=False)
    except OSError:
        return False
    return True
