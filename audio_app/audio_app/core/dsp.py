"""Briques de base du traitement du signal : STFT, STFT inverse, masque doux.

Idée de la STFT : un son change avec le temps, donc on le découpe en petites
fenêtres qui se chevauchent et on fait une transformée de Fourier sur chacune.
Résultat : une matrice (fréquences x temps) de nombres complexes. Module = énergie
à cette fréquence, phase = « position » de l'onde.
"""
from scipy import signal as sps

EPS = 1e-10  # petite constante pour éviter les divisions par zéro


def calculer_stft(x, fs, n_fft=2048, hop=1024):
    """STFT avec fenêtre de Hann. hop = décalage entre deux fenêtres."""
    _, _, Z = sps.stft(x, fs=fs, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
    return Z


def inverser_stft(Z, fs, n_fft, hop, longueur):
    """STFT inverse : reconstruit le signal temporel à partir de la matrice Z.
    On recoupe/complète pour retrouver exactement la longueur d'origine."""
    import numpy as np

    _, x = sps.istft(Z, fs=fs, nperseg=n_fft, noverlap=n_fft - hop, window="hann")
    if len(x) < longueur:
        x = np.pad(x, (0, longueur - len(x)))
    return x[:longueur]


def masque_doux(a, b, puissance=2):
    """Masque doux (type filtre de Wiener) : part de 'a' dans le mélange a+b, entre 0 et 1.
    Appliqué à la STFT : 1 = on garde tout, 0 = on supprime."""
    a_p, b_p = a ** puissance, b ** puissance
    return a_p / (a_p + b_p + EPS)
