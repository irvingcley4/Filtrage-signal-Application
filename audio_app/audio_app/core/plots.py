"""Graphiques enregistrés en PNG (spectrogrammes, segmentation).

On utilise l'API objet de Matplotlib (Figure + canvas Agg) et non pyplot : pyplot garde
un état global et n'est pas fait pour être appelé depuis un thread de calcul.
"""
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy import signal as sps

from .podcast import COULEURS_SEGMENTS


def tracer_spectrogrammes(signaux, fs, fichier_png, titre=""):
    """Trace un spectrogramme par signal (dict {nom: signal}), côte à côte.
    Axe x = temps, axe y = fréquence, couleur = énergie en décibels.
    Le bruit de vent apparaît comme une bande claire en bas, le souffle comme un voile
    uniforme ; ils doivent disparaître après traitement."""
    n = len(signaux)
    fig = Figure(figsize=(5.5 * n, 4))
    FigureCanvasAgg(fig)
    axes = np.atleast_1d(fig.subplots(1, n, sharey=True))
    for ax, (nom, x) in zip(axes, signaux.items()):
        f, t, Z = sps.stft(x, fs=fs, nperseg=1024, noverlap=768)
        db = 20 * np.log10(np.abs(Z) + 1e-8)
        ax.pcolormesh(t, f, db, vmin=db.max() - 80, vmax=db.max(),
                      shading="auto", cmap="magma")
        ax.set_title(nom)
        ax.set_xlabel("Temps (s)")
        ax.set_ylim(0, min(fs / 2, 8000))        # la voix vit sous ~8 kHz
    axes[0].set_ylabel("Fréquence (Hz)")
    if titre:
        fig.suptitle(titre)
    fig.tight_layout()
    fig.savefig(str(fichier_png), dpi=120)
    return Path(fichier_png)


def tracer_segmentation(env_db, pas, etiquettes, fichier_png):
    """Trace l'enveloppe d'énergie avec des zones colorées par type de segment."""
    fig = Figure(figsize=(11, 3))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    t = np.arange(len(env_db)) * pas
    ax.plot(t, env_db, color="black", linewidth=0.6)
    for i, e in enumerate(etiquettes):
        ax.axvspan(i, i + 1, color=COULEURS_SEGMENTS[e], alpha=0.35)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Niveau (dB)")
    ax.set_title("Segmentation  (bleu : parole, orange : musique, violet : mixte, gris : silence)")
    fig.tight_layout()
    fig.savefig(str(fichier_png), dpi=120)
    return Path(fichier_png)
