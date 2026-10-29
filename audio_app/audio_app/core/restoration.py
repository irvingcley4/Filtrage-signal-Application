"""MODE C : restauration vocale = filtre passe-haut Butterworth + soustraction spectrale."""
import numpy as np
from scipy import signal as sps
from scipy.ndimage import uniform_filter1d

from .dsp import EPS, calculer_stft, inverser_stft


def choisir_coupure_dynamique(x, fs):
    """Fréquence de coupure du passe-haut, adaptée au signal (entre 100 et 120 Hz).
    On mesure la part d'énergie sous 150 Hz (spectre de Welch) : plus il y a de vent,
    plus cette part est grande, et plus on coupe haut."""
    f, psd = sps.welch(x, fs=fs, nperseg=min(4096, len(x)))
    energie_basse = psd[(f >= 20) & (f <= 150)].sum()
    energie_totale = psd[f >= 20].sum() + EPS
    ratio = energie_basse / energie_totale
    return 100.0 + 20.0 * float(np.clip(ratio / 0.3, 0.0, 1.0))


def filtre_passe_haut(x, fs, fc, ordre=4):
    """Filtre de Butterworth passe-haut : supprime ce qui est sous fc (Hz).
    - Butterworth = réponse plate dans la bande passante (ne déforme pas la voix).
    - 'sos' (sections d'ordre 2) = forme numériquement stable.
    - sosfiltfilt = filtre aller-retour : aucun décalage de phase."""
    sos = sps.butter(ordre, fc, btype="highpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x)


def estimer_profil_bruit(mag, duree_bruit_s, fs, hop):
    """Estime le spectre moyen du bruit.
    On cherche automatiquement la fenêtre la plus calme du fichier (là où il y a
    sans doute que du bruit), au lieu de supposer qu'elle est au début."""
    n_trames = max(2, int(duree_bruit_s * fs / hop))
    energie = (mag ** 2).sum(axis=0)                           # énergie de chaque trame
    energie_lissee = uniform_filter1d(energie, size=n_trames)  # moyenne glissante
    centre = int(np.argmin(energie_lissee))                    # centre de la zone la + calme
    debut = max(0, min(centre - n_trames // 2, mag.shape[1] - n_trames))
    return mag[:, debut: debut + n_trames].mean(axis=1, keepdims=True)


def soustraction_spectrale(x, fs, alpha=2.0, beta=0.05, duree_bruit_s=0.5):
    """Débruitage par soustraction spectrale.
    1) STFT du signal.
    2) Profil de bruit = spectre moyen d'une zone calme.
    3) On soustrait alpha * profil au module de chaque trame
       (alpha > 1 = sur-soustraction, car le bruit réel fluctue autour de sa moyenne).
    4) Plancher beta * module : on ne descend jamais à zéro, sinon on entend des
       "bulles" métalliques (bruit musical). beta garde un petit voile naturel.
    5) On lisse le gain dans le temps (autre astuce anti-bruit musical).
    6) On garde la phase d'origine et on repasse en temporel (STFT inverse)."""
    n_fft, hop = 1024, 256
    Z = calculer_stft(x, fs, n_fft, hop)
    mag = np.abs(Z)
    profil = estimer_profil_bruit(mag, duree_bruit_s, fs, hop)

    propre = np.maximum(mag - alpha * profil, beta * mag)      # soustraction + plancher
    gain = propre / (mag + EPS)                                # facteur d'atténuation 0..1
    gain = uniform_filter1d(gain, size=3, axis=1)              # lissage temporel
    return inverser_stft(Z * gain, fs, n_fft, hop, len(x))


def restaurer_voix(x, fs, alpha=2.0, info=None, rappel=None):
    """Chaîne complète du mode C : passe-haut dynamique, puis soustraction spectrale.
    On filtre d'abord les graves du vent : sinon leur énergie fausserait le profil de bruit.

    info   : dict optionnel ; reçoit « coupure_hz », la coupure réellement utilisée
             (remplace le `print` du script d'origine).
    rappel : fonction optionnelle rappel(statut, fraction 0..1) appelée entre les étapes."""
    if rappel:
        rappel("Analyse de la fréquence de coupure…", 0.0)
    fc = choisir_coupure_dynamique(x, fs)
    if info is not None:
        info["coupure_hz"] = fc
    if rappel:
        rappel("Filtrage passe-haut…", 0.25)
    filtre = filtre_passe_haut(x, fs, fc)
    if rappel:
        rappel("Soustraction spectrale…", 0.5)
    return soustraction_spectrale(filtre, fs, alpha=alpha)
