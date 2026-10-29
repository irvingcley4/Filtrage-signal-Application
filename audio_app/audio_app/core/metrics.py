"""Métriques numériques : SNR mesuré (avec référence), SNR estimé (sans), parts d'énergie."""
import numpy as np

from .dsp import EPS


def snr_db(reference, estimation):
    """SNR (Rapport Signal sur Bruit) en dB, quand on CONNAÎT le signal propre.
        SNR = 10 * log10( énergie du signal / énergie de l'erreur )
    Erreur = reference - estimation. +10 dB = erreur 10 fois plus faible que le signal."""
    n = min(len(reference), len(estimation))
    ref, est = reference[:n], estimation[:n]
    erreur = ref - est
    return 10 * np.log10((np.sum(ref ** 2) + EPS) / (np.sum(erreur ** 2) + EPS))


def snr_estime(x, fs):
    """SNR ESTIMÉ quand on n'a PAS de signal propre (cas d'un vrai enregistrement).
    On découpe en trames de 20 ms, puis :
      - bruit  ~ puissance moyenne des 10 % de trames les plus faibles (les pauses)
      - signal ~ puissance moyenne des 30 % de trames les plus fortes, moins le bruit
    C'est une approximation : elle sert à COMPARER avant/après, pas à donner une valeur absolue."""
    taille = int(0.02 * fs)
    n = len(x) // taille
    puissances = np.sort((x[: n * taille].reshape(n, taille) ** 2).mean(axis=1))
    p_bruit = puissances[: max(1, n // 10)].mean() + EPS
    p_signal = puissances[-max(1, int(0.3 * n)):].mean()
    return 10 * np.log10(max(p_signal - p_bruit, EPS) / p_bruit)


def parts_energie(voix, fond):
    """Parts d'énergie (en %) de la voix et du fond estimés, dans le total des deux.
    Ce n'est PAS une mesure de qualité de séparation : seulement une répartition d'énergie."""
    e_v, e_f = float(np.sum(voix ** 2)), float(np.sum(fond ** 2))
    total = e_v + e_f + EPS
    return 100 * e_v / total, 100 * e_f / total
