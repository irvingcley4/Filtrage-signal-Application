"""Séparation voix / instrumental par Demucs (apprentissage profond).

Conception :
  * tout ce qui touche à PyTorch/Demucs est isolé dans quelques petites fonctions
    (`_importer_demucs`, `_charger_modele`, `_executer_bloc`, `detecter_peripherique`),
    importées PARESSEUSEMENT : l'application démarre et reste utilisable sans torch ;
  * le reste (normalisation, découpage en blocs, fondus, assemblage, validation) est du
    NumPy pur, donc testable sans Demucs ;
  * traitement bloc par bloc (30 s, recouvrement 3 s, fondu linéaire) : permet une
    progression RÉELLE et une annulation entre deux blocs ; Demucs lui-même découpe
    encore chaque bloc en segments avec recouvrement (comportement standard).

Les sorties sont en STÉRÉO à la fréquence du modèle (44100 Hz pour htdemucs).
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Optional

import numpy as np

from .environnement import demucs_disponible, message_installation_demucs
from .erreurs import AudioError, DependanceManquanteError
from .resultats import Rappel, signaler

# --- paramètres (modifiables ici) -------------------------------------------------------
MODELES = {
    "htdemucs": "htdemucs — modèle par défaut (recommandé)",
    "htdemucs_ft": "htdemucs_ft — qualité potentiellement meilleure, environ 4 fois plus lent",
}
MODELE_PAR_DEFAUT = "htdemucs"
FREQUENCE_DEMUCS = 44100
PARAMETRES_DEMUCS = {
    "shifts": 1,          # 1 passe décalée aléatoirement (0 = plus rapide, plus de 1 = plus lent)
    "overlap": 0.25,      # recouvrement entre segments internes de Demucs
    "segment": None,      # None = valeur du modèle ; réduite automatiquement si la mémoire manque
    "num_workers": 0,     # tâches parallèles de Demucs (0 = aucune surcharge de processus)
}
SEGMENT_REDUIT_S = 4.0    # utilisé en secours après une erreur de mémoire
BLOC_S = 30.0             # longueur d'un bloc traité
RECOUVREMENT_BLOCS_S = 3.0


# ======================================================================================
# Accès à PyTorch / Demucs (imports paresseux)
# ======================================================================================
def verifier_demucs() -> None:
    """Lève DependanceManquanteError (avec les commandes d'installation) si Demucs est absent."""
    if not demucs_disponible():
        raise DependanceManquanteError(message_installation_demucs())


def _importer_demucs():
    """Retourne (get_model, apply_model, nom_du_paquet). Essaie « demucs » puis le fork
    « demucs_infer » (même API, compatible PyTorch récent)."""
    derniere = ""
    for paquet in ("demucs", "demucs_infer"):
        try:
            pretrained = importlib.import_module(f"{paquet}.pretrained")
            apply = importlib.import_module(f"{paquet}.apply")
            return pretrained.get_model, apply.apply_model, paquet
        except ImportError as exc:
            derniere = f"{paquet} : {exc}"
    raise DependanceManquanteError(message_installation_demucs(derniere))


def detecter_peripherique(preference: str = "auto"):
    """Retourne (« cuda » ou « cpu », nom lisible). CUDA seulement s'il est réellement utilisable."""
    import torch

    if preference != "cpu" and torch.cuda.is_available():
        try:
            return "cuda", f"GPU NVIDIA — {torch.cuda.get_device_name(0)}"
        except Exception:                      # pilote défaillant : on retombe sur le CPU
            pass
    return "cpu", "CPU (processeur)"


def modele_probablement_en_cache() -> bool:
    """Heuristique : le dossier de cache de PyTorch contient-il déjà des modèles (.th) ?
    Sert seulement à prévenir qu'une connexion Internet sera nécessaire au premier lancement."""
    try:
        import torch

        dossier = Path(torch.hub.get_dir()) / "checkpoints"
        return dossier.is_dir() and any(dossier.glob("*.th"))
    except Exception:
        return False


def _charger_modele(nom: str):
    get_model, _, _ = _importer_demucs()
    modele = get_model(nom)
    modele.eval()
    return modele


def _executer_bloc(modele, audio: np.ndarray, device: str, shifts: int, overlap: float,
                   segment: Optional[float], num_workers: int) -> np.ndarray:
    """Applique le modèle à un bloc (C, T) normalisé -> tableau (S, C, T) (une ligne par source)."""
    import torch

    _, apply_model, _ = _importer_demucs()
    tenseur = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
    options = dict(shifts=shifts, split=True, overlap=overlap, progress=False,
                   device=device, num_workers=num_workers)
    if segment is not None:
        options["segment"] = segment
    with torch.no_grad():
        sortie = apply_model(modele, tenseur, **options)
    return sortie[0].detach().cpu().numpy()


# ======================================================================================
# Erreurs
# ======================================================================================
def _categorie_erreur(exc: BaseException) -> Optional[str]:
    """« memoire », « cuda » ou None, d'après le type et le texte de l'exception."""
    texte = str(exc).lower()
    if isinstance(exc, MemoryError) or "out of memory" in texte or "can't allocate memory" in texte \
            or "defaultcpuallocator" in texte:
        return "memoire"
    if "cuda" in texte or "cudnn" in texte or "cublas" in texte:
        return "cuda"
    return None


def _erreur_chargement(exc: BaseException, nom_modele: str) -> AudioError:
    """Traduit une exception de chargement/téléchargement du modèle en message lisible."""
    texte = f"{type(exc).__name__}: {exc}".lower()
    reseau = any(m in texte for m in ("urlerror", "urlopen", "getaddrinfo", "name or service",
                                      "connection", "timed out", "timeout", "http error",
                                      "temporary failure", "network"))
    if reseau:
        return AudioError(
            f"Le modèle Demucs « {nom_modele} » n'a pas pu être téléchargé. Le premier lancement "
            "nécessite une connexion Internet (le modèle est ensuite conservé dans le cache de PyTorch). "
            "Vérifiez votre connexion puis réessayez.")
    if _categorie_erreur(exc) == "memoire":
        return AudioError("Mémoire insuffisante pour charger le modèle Demucs. Fermez d'autres "
                          "applications ou essayez le modèle htdemucs (plus léger que htdemucs_ft).")
    return AudioError(f"Le modèle Demucs « {nom_modele} » n'a pas pu être chargé "
                      "(nom inconnu ou fichier corrompu ; voir les détails techniques).")


# ======================================================================================
# Découpage en blocs (NumPy pur)
# ======================================================================================
def plan_blocs(n: int, fs: int, bloc_s: float = BLOC_S, recouvrement_s: float = RECOUVREMENT_BLOCS_S):
    """Liste de blocs (début, fin) couvrant [0, n) ; blocs consécutifs recouverts de `recouvrement_s`.
    Un morceau de moins d'environ 45 s est traité d'un seul bloc."""
    bloc, ov = int(bloc_s * fs), int(recouvrement_s * fs)
    if n <= int(1.5 * bloc):
        return [(0, n)]
    pas = bloc - ov
    blocs, debut = [], 0
    while True:
        fin = min(debut + bloc, n)
        blocs.append((debut, fin))
        if fin >= n:
            break
        debut += pas
    if len(blocs) > 1 and blocs[-1][1] - blocs[-1][0] < 2 * ov:   # dernier bloc trop court : on le fusionne
        blocs.pop()
        blocs[-1] = (blocs[-1][0], n)
    return blocs


def poids_bloc(longueur: int, fondu_debut: int, fondu_fin: int) -> np.ndarray:
    """Poids d'un bloc : fondu linéaire en entrée et en sortie. Deux blocs voisins ont des poids
    complémentaires (somme = 1) sur leur zone de recouvrement."""
    w = np.ones(longueur, dtype=np.float32)
    if fondu_debut > 0:
        w[:fondu_debut] = (np.arange(fondu_debut) + 0.5) / fondu_debut
    if fondu_fin > 0:
        w[longueur - fondu_fin:] *= (fondu_fin - np.arange(fondu_fin) - 0.5) / fondu_fin
    return w


def valider_sorties(voix: np.ndarray, instrumental: np.ndarray, n_attendu: int) -> None:
    """Vérifie que les stems exportables sont non vides, finis, de même forme et de la bonne durée."""
    for nom, s in (("vocals", voix), ("instrumental", instrumental)):
        if s.ndim != 2 or s.shape[1] == 0:
            raise AudioError(f"La source « {nom} » produite par Demucs est vide.")
        if not np.all(np.isfinite(s)):
            raise AudioError(f"La source « {nom} » produite par Demucs contient des valeurs invalides.")
    if voix.shape != instrumental.shape or voix.shape[1] != n_attendu:
        raise AudioError("Les sources produites par Demucs n'ont pas des dimensions cohérentes "
                         f"(attendu {n_attendu} échantillons, reçu {voix.shape} et {instrumental.shape}).")


# ======================================================================================
# Séparateur
# ======================================================================================
class SeparateurDemucs:
    """Charge le modèle une fois, puis sépare un ou plusieurs signaux stéréo (2, N).

    `rappel(statut, progression)` : progression = fraction réelle (blocs traités) ou None si inconnue.
    Il peut lever TraitementAnnule : le calcul s'arrête alors entre deux blocs.
    """

    def __init__(self, nom_modele: str = MODELE_PAR_DEFAUT, peripherique: str = "auto",
                 rappel: Optional[Rappel] = None, **parametres):
        self.nom_modele = nom_modele
        self._preference = peripherique
        self._rappel = rappel
        self._parametres = {**PARAMETRES_DEMUCS, **parametres}
        self.modele = None
        self.device = ""
        self.nom_peripherique = ""
        self.samplerate = FREQUENCE_DEMUCS
        self.sources: list[str] = []
        self.avertissements: list[str] = []
        self._threads_avant: Optional[int] = None

    # ------------------------------------------------------------------
    def infos(self) -> dict:
        return {"modele": self.nom_modele, "peripherique": self.nom_peripherique, "device": self.device}

    def charger(self) -> None:
        """Vérifie les dépendances, choisit le périphérique et charge le modèle (une seule fois)."""
        if self.modele is not None:
            return
        verifier_demucs()
        self.device, self.nom_peripherique = detecter_peripherique(self._preference)
        message = "Chargement du modèle de séparation…"
        if not modele_probablement_en_cache():
            message += " (premier lancement : téléchargement du modèle, connexion Internet requise)"
        signaler(self._rappel, message, None)
        try:
            modele = _charger_modele(self.nom_modele)
        except DependanceManquanteError:
            raise
        except Exception as exc:
            raise _erreur_chargement(exc, self.nom_modele) from exc
        sources = list(getattr(modele, "sources", []))
        if "vocals" not in sources:
            raise AudioError(f"Le modèle « {self.nom_modele} » ne produit pas de source « vocals » : "
                             "il ne convient pas à la séparation voix / instrumental.")
        self.modele, self.sources = modele, sources
        self.samplerate = int(getattr(modele, "samplerate", FREQUENCE_DEMUCS))
        if self.device == "cpu":
            self._limiter_threads_cpu()

    def liberer(self) -> None:
        """Libère le modèle (et la mémoire GPU) et restaure le nombre de threads de PyTorch."""
        self.modele = None
        try:
            import torch

            if self._threads_avant is not None:
                torch.set_num_threads(self._threads_avant)
                self._threads_avant = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _limiter_threads_cpu(self) -> None:
        """Laisse un cœur libre pour l'interface pendant le calcul sur CPU."""
        try:
            import torch

            self._threads_avant = torch.get_num_threads()
            torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _executer(self, bloc: np.ndarray) -> np.ndarray:
        """Exécute le modèle sur un bloc, avec repli GPU -> CPU et réduction de segment si mémoire courte."""
        while True:
            try:
                return _executer_bloc(self.modele, bloc, self.device,
                                      self._parametres["shifts"], self._parametres["overlap"],
                                      self._parametres["segment"], self._parametres["num_workers"])
            except Exception as exc:
                categorie = _categorie_erreur(exc)
                if categorie is None:
                    raise
                if self.device == "cuda":
                    self._basculer_sur_cpu(f"erreur {'de mémoire ' if categorie == 'memoire' else ''}GPU")
                elif categorie == "memoire" and self._parametres["segment"] is None:
                    self._parametres["segment"] = SEGMENT_REDUIT_S
                    self.avertissements.append("Mémoire insuffisante : traitement relancé avec des "
                                               "segments plus courts (plus lent).")
                    signaler(self._rappel, "Mémoire insuffisante : nouvelle tentative avec des segments plus courts…", None)
                elif categorie == "memoire":
                    raise AudioError("Mémoire insuffisante pour traiter ce fichier, même avec des segments "
                                     "courts. Fermez d'autres applications ou utilisez un fichier plus court.") from exc
                else:
                    raise

    def _basculer_sur_cpu(self, raison: str) -> None:
        try:
            self.modele.to("cpu")
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self.device, self.nom_peripherique = "cpu", "CPU (processeur, repli après une erreur GPU)"
        self.avertissements.append(f"Le GPU a provoqué une {raison} : le calcul a continué sur le processeur.")
        signaler(self._rappel, f"{raison.capitalize()} : le calcul continue sur le processeur…", None)
        self._limiter_threads_cpu()

    def separer(self, audio: np.ndarray, rappel: Optional[Rappel] = None):
        """audio : (2, N) float à la fréquence `self.samplerate`. Retourne (voix, instrumental), chacun (2, N).
        L'instrumental est la somme des autres sources du modèle (drums + bass + other), ou la sortie
        « no_vocals » si le modèle en fournit réellement une."""
        rappel = rappel if rappel is not None else self._rappel
        if audio.ndim != 2 or audio.shape[0] != 2:
            raise AudioError("Demucs attend un signal stéréo (2 canaux).")
        self.charger()
        n = audio.shape[1]
        if n == 0 or not np.all(np.isfinite(audio)):
            raise AudioError("Le signal audio est vide ou corrompu.")

        # Normalisation de référence de Demucs : (x - moyenne) / écart-type du signal mono, puis
        # sources * écart-type. Elle est inversée à la sortie : le niveau d'origine est conservé.
        ref = audio.mean(axis=0)
        moyenne, ecart = float(ref.mean()), float(ref.std())
        if ecart < 1e-8:
            raise AudioError("Le signal est silencieux : il n'y a rien à séparer.")
        normalise = ((audio - moyenne) / ecart).astype(np.float32)

        i_voix = self.sources.index("vocals")
        autres = [i for i, nom in enumerate(self.sources) if nom != "vocals"]
        if "no_vocals" in self.sources:
            autres = [self.sources.index("no_vocals")]

        voix = np.zeros((2, n), dtype=np.float32)
        instrumental = np.zeros((2, n), dtype=np.float32)
        blocs = plan_blocs(n, self.samplerate)
        for k, (debut, fin) in enumerate(blocs):
            if len(blocs) == 1:
                signaler(rappel, f"Analyse de la musique et extraction de la voix… (périphérique : "
                                 f"{self.nom_peripherique})", None)
            else:
                signaler(rappel, f"Analyse de la musique et extraction de la voix… bloc {k + 1}/{len(blocs)} "
                                 f"(périphérique : {self.nom_peripherique})", k / len(blocs))
            sources = self._executer(normalise[:, debut:fin]) * ecart
            if sources.ndim != 3 or sources.shape[0] != len(self.sources) or sources.shape[2] != fin - debut:
                raise AudioError("Sortie inattendue de Demucs (dimensions des sources incohérentes).")
            fondu_in = blocs[k - 1][1] - debut if k > 0 else 0
            fondu_out = fin - blocs[k + 1][0] if k < len(blocs) - 1 else 0
            w = poids_bloc(fin - debut, fondu_in, fondu_out)
            voix[:, debut:fin] += w * sources[i_voix]
            instrumental[:, debut:fin] += w * sources[autres].sum(axis=0)
        signaler(rappel, "Création de l'instrumental…", 0.99 if len(blocs) > 1 else None)
        valider_sorties(voix, instrumental, n)
        return voix, instrumental
