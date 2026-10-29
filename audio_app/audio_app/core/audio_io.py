"""Entrées/sorties audio : lecture (WAV direct, autres formats via FFmpeg), écriture WAV.

Différence avec le script d'origine : les erreurs sont des exceptions (AudioError et
dérivées) au lieu de `sys.exit()`, qui tuerait silencieusement un thread de calcul.
"""
from __future__ import annotations

import json
import subprocess
import wave
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import signal as sps
from scipy.io import wavfile

from .dsp import EPS
from .environnement import chemin_ffmpeg, chemin_ffprobe, options_subprocess
from .erreurs import (AudioTropCourtError, FfmpegIntrouvableError,
                      FichierInvalideError)

EXTENSIONS_AUDIO = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus", ".wma", ".mpeg", ".mpg")


@dataclass(frozen=True)
class InfosAudio:
    """Métadonnées d'un fichier audio, lues sans décoder tout le signal quand c'est possible."""
    chemin: Path
    duree_s: float
    frequence: int
    taille_octets: int

    @property
    def nom(self) -> str:
        return self.chemin.name


def _lire_avec_ffmpeg(chemin: Path, sr: int, canaux: int = 1):
    """Décode n'importe quel format (mp3, m4a, ...) via ffmpeg, en float32 (mono par défaut,
    ou 2 canaux : tableau (2, N)). ffmpeg écrit les échantillons bruts sur sa sortie standard."""
    ffmpeg = chemin_ffmpeg()
    if ffmpeg is None:
        raise FfmpegIntrouvableError(
            "FFmpeg est requis pour lire ce format audio. Installez-le (voir le README) "
            "puis relancez l'application, ou convertissez le fichier en WAV.")
    cmd = [ffmpeg, "-v", "error", "-i", str(chemin),
           "-f", "f32le", "-ac", str(canaux), "-ar", str(sr), "-"]
    try:
        brut = subprocess.run(cmd, capture_output=True, check=True, **options_subprocess()).stdout
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip().splitlines()
        raise FichierInvalideError(
            "Ce fichier n'a pas pu être décodé par FFmpeg (fichier corrompu ou format non audio)."
            + (f" Détail : {detail[-1]}" if detail else "")) from exc
    x = np.frombuffer(brut, dtype=np.float32).copy()
    if x.size == 0:
        raise FichierInvalideError("Le fichier ne contient aucun échantillon audio.")
    if canaux == 2:
        x = x[: x.size // 2 * 2].reshape(-1, 2).T.copy()      # (N, 2) entrelacé -> (2, N)
    return x, sr


def charger_audio(chemin, sr_cible: Optional[int] = None):
    """Charge un fichier audio -> (signal mono float entre -1 et 1, fréquence d'échantillonnage).
    sr_cible : si donné, rééchantillonne à cette fréquence (accélère les modes A et B)."""
    chemin = Path(chemin)
    if not chemin.is_file():
        raise FichierInvalideError(f"Fichier introuvable : {chemin}")

    x, fs = None, None
    if chemin.suffix.lower() == ".wav":
        try:
            fs, data = wavfile.read(str(chemin))
            if data.dtype == np.int16:
                x = data / 32768.0
            elif data.dtype == np.int32:
                x = data / 2147483648.0
            elif data.dtype == np.uint8:
                x = (data - 128) / 128.0
            else:
                x = data.astype(np.float64)
            if x.ndim > 1:             # stéréo -> mono (moyenne des canaux)
                x = x.mean(axis=1)
        except ValueError:             # wav exotique (ex. 24 bits) : on laisse ffmpeg s'en charger
            x = None
    if x is None:
        x, fs = _lire_avec_ffmpeg(chemin, sr_cible or 44100)

    x = x.astype(np.float64)
    if x.size == 0:
        raise FichierInvalideError("Le fichier ne contient aucun échantillon audio.")
    if sr_cible and fs != sr_cible:    # rééchantillonnage polyphasé (rapide et propre)
        g = gcd(int(fs), int(sr_cible))
        x = sps.resample_poly(x, sr_cible // g, fs // g)
        fs = sr_cible
    return x, int(fs)


def charger_audio_stereo(chemin, sr_cible: int):
    """Charge un fichier en STÉRÉO float32, forme (2, N), à la fréquence `sr_cible` (Demucs : 44100 Hz).
    Aucune normalisation. Mono -> les deux canaux sont identiques ; plus de 2 canaux -> les deux premiers.
    Le fichier source n'est jamais modifié."""
    chemin = Path(chemin)
    if not chemin.is_file():
        raise FichierInvalideError(f"Fichier introuvable : {chemin}")
    audio, fs = None, None
    if chemin.suffix.lower() == ".wav":
        try:
            fs, data = wavfile.read(str(chemin))
            if data.dtype == np.int16:
                x = data / 32768.0
            elif data.dtype == np.int32:
                x = data / 2147483648.0
            elif data.dtype == np.uint8:
                x = (data - 128) / 128.0
            else:
                x = data.astype(np.float64)
            if x.ndim == 1:
                x = np.stack([x, x], axis=1)
            elif x.shape[1] == 1:
                x = np.repeat(x, 2, axis=1)
            else:
                x = x[:, :2]
            audio = x.T
        except ValueError:                 # wav exotique : FFmpeg
            audio = None
    if audio is None:
        audio, fs = _lire_avec_ffmpeg(chemin, sr_cible, canaux=2)
    if audio.size == 0:
        raise FichierInvalideError("Le fichier ne contient aucun échantillon audio.")
    if not np.all(np.isfinite(audio)):
        raise FichierInvalideError("Le fichier audio est corrompu (valeurs invalides).")
    audio = reechantillonner(audio, int(fs), int(sr_cible))
    return np.ascontiguousarray(audio, dtype=np.float32), int(sr_cible)


def reechantillonner(audio, fs: int, fs_cible: int):
    """Rééchantillonnage polyphasé le long du dernier axe (tableau (C, N) ou (N,))."""
    if fs == fs_cible:
        return audio
    g = gcd(int(fs), int(fs_cible))
    return sps.resample_poly(audio, fs_cible // g, fs // g, axis=-1)


def verifier_wav_exporte(chemin, nb_echantillons: int, fs: int, canaux: int) -> None:
    """Relit l'en-tête du WAV écrit : durée, fréquence et nombre de canaux doivent correspondre."""
    try:
        with wave.open(str(chemin), "rb") as w:
            ok = (w.getnframes() == nb_echantillons and w.getframerate() == fs
                  and w.getnchannels() == canaux)
    except (wave.Error, EOFError, OSError) as exc:
        raise FichierInvalideError(f"Le fichier exporté {Path(chemin).name} est illisible.") from exc
    if not ok:
        raise FichierInvalideError(f"Le fichier exporté {Path(chemin).name} n'a pas les dimensions attendues.")


def sauvegarder_audio(chemin, x, fs) -> None:
    """Écrit un .wav 16 bits. On normalise le pic pour éviter l'écrêtage (saturation)."""
    pic = np.max(np.abs(x)) + EPS
    if pic > 0.99:
        x = x / pic * 0.99
    wavfile.write(str(chemin), int(fs), (x * 32767).astype(np.int16))


def verifier_duree_minimale(x, fs, minimum_s: float, message: str) -> None:
    """Lève AudioTropCourtError si le signal dure moins de `minimum_s` secondes."""
    if len(x) / fs < minimum_s:
        raise AudioTropCourtError(message)


# --------------------------------------------------------------------------
# Métadonnées (pour l'affichage avant traitement)
# --------------------------------------------------------------------------
def _infos_wav(chemin: Path):
    """Durée et fréquence d'un WAV PCM via l'en-tête (rapide). None si format non PCM."""
    try:
        with wave.open(str(chemin), "rb") as w:
            fs = w.getframerate()
            return (w.getnframes() / fs, fs) if fs > 0 else None
    except (wave.Error, EOFError):
        return None


def _infos_ffprobe(chemin: Path):
    """Durée et fréquence via ffprobe. None si ffprobe est absent ou ne donne pas de durée."""
    ffprobe = chemin_ffprobe()
    if ffprobe is None:
        return None
    cmd = [ffprobe, "-v", "error", "-select_streams", "a:0",
           "-show_entries", "stream=sample_rate,duration", "-show_entries", "format=duration",
           "-of", "json", str(chemin)]
    try:
        sortie = subprocess.run(cmd, capture_output=True, check=True, **options_subprocess()).stdout
        donnees = json.loads(sortie.decode("utf-8", errors="replace"))
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise FichierInvalideError("Ce fichier n'est pas un fichier audio lisible.") from exc
    flux = donnees.get("streams") or []
    if not flux:
        raise FichierInvalideError("Aucune piste audio n'a été trouvée dans ce fichier.")
    try:
        duree = float(flux[0].get("duration") or donnees.get("format", {}).get("duration"))
        fs = int(flux[0]["sample_rate"])
    except (TypeError, ValueError, KeyError):
        return None
    return duree, fs


def lire_infos(chemin) -> InfosAudio:
    """Nom, durée, fréquence et taille d'un fichier audio, avec validation du contenu.
    Ordre : en-tête WAV -> ffprobe -> décodage complet (dernier recours, nécessite FFmpeg)."""
    chemin = Path(chemin)
    if not chemin.is_file():
        raise FichierInvalideError(f"Fichier introuvable : {chemin}")
    taille = chemin.stat().st_size
    if taille == 0:
        raise FichierInvalideError("Ce fichier est vide.")

    infos = _infos_wav(chemin) if chemin.suffix.lower() == ".wav" else None
    if infos is None:
        infos = _infos_ffprobe(chemin)
    if infos is None:
        x, fs = charger_audio(chemin)
        infos = (len(x) / fs, fs)
    duree, fs = infos
    if duree <= 0:
        raise FichierInvalideError("Ce fichier ne contient aucun échantillon audio.")
    return InfosAudio(chemin, float(duree), int(fs), int(taille))


# --------------------------------------------------------------------------
# Lecture dans l'application
# --------------------------------------------------------------------------
def convertir_pour_lecture(source, dossier_temp) -> Path:
    """Convertit un fichier en WAV temporaire pour QMediaPlayer (formats non pris en charge
    par le système). Le fichier source n'est jamais modifié : on écrit dans `dossier_temp`."""
    ffmpeg = chemin_ffmpeg()
    if ffmpeg is None:
        raise FfmpegIntrouvableError("FFmpeg est requis pour lire ce format audio.")
    source, dossier_temp = Path(source), Path(dossier_temp)
    sortie = dossier_temp / f"lecture_{source.stem}.wav"
    cmd = [ffmpeg, "-v", "error", "-y", "-i", str(source), "-vn", "-ar", "44100", str(sortie)]
    try:
        subprocess.run(cmd, capture_output=True, check=True, **options_subprocess())
    except subprocess.CalledProcessError as exc:
        raise FichierInvalideError("La conversion du fichier pour l'écoute a échoué.") from exc
    return sortie
