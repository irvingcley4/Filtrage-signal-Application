"""Démonstrations : signaux synthétiques dont on connaît la vérité -> SNR réellement mesuré.

Les générateurs reprennent exactement ceux du script d'origine ; les fonctions
`executer_*` exécutent les VRAIS algorithmes de restauration et de séparation.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import signal as sps

from .audio_io import sauvegarder_audio
from .dsp import EPS
from .metrics import parts_energie, snr_db
from .plots import tracer_spectrogrammes
from .restoration import restaurer_voix
from .resultats import Metrique, Rappel, ResultatTraitement, signaler
from .separation import separer_voix_fond, verifier_sklearn


def voix_synthetique(fs, duree, f0_fonction, silence_debut=0.0):
    """Fabrique une 'voix' artificielle : un son harmonique (f0 + multiples) dont la hauteur
    varie, haché par une enveloppe qui imite les syllabes (~2,5 par seconde)."""
    t = np.arange(int(duree * fs)) / fs
    phase = 2 * np.pi * np.cumsum(f0_fonction(t)) / fs
    son = sum(np.sin(h * phase) / h for h in range(1, 16))
    sos = sps.butter(4, 3500, btype="lowpass", fs=fs, output="sos")
    son = sps.sosfilt(sos, son)
    enveloppe = np.clip(np.sin(2 * np.pi * 2.5 * t), 0, None) ** 0.7
    enveloppe[t < silence_debut] = 0.0
    return 0.4 * son * enveloppe / (np.max(np.abs(son)) + EPS)


def generer_demo_restauration():
    """Voix synthétique + grondement de vent + souffle. Retourne (propre, bruité, fs)."""
    fs, duree = 16000, 6.0
    rng = np.random.default_rng(0)
    propre = voix_synthetique(fs, duree, lambda t: 120 + 40 * np.sin(2 * np.pi * 0.4 * t),
                              silence_debut=0.7)
    sos = sps.butter(4, 90, btype="lowpass", fs=fs, output="sos")
    vent = sps.sosfilt(sos, rng.standard_normal(len(propre)))
    vent = 0.3 * vent / np.max(np.abs(vent))                      # grondement grave
    souffle = 0.02 * rng.standard_normal(len(propre))             # bruit blanc
    return propre, propre + vent + souffle, fs


def generer_demo_musique():
    """Voix synthétique + boucle instrumentale de 2 s répétée.
    Retourne (voix, instrumental, mélange, fs)."""
    fs, duree = 22050, 14.0
    # --- Instrumental : une boucle de 2 s répétée -> exactement ce que REPET-SIM exploite ---
    rng = np.random.default_rng(1)
    boucle = np.zeros(2 * fs)
    tb = np.arange(int(0.15 * fs)) / fs
    for debut_s in (0.0, 0.5, 1.0, 1.5):                          # grosse caisse (4 temps)
        i = int(debut_s * fs)
        boucle[i: i + len(tb)] += 0.6 * np.sin(2 * np.pi * 55 * tb) * np.exp(-tb * 25)
    for debut_s in (0.25, 0.75, 1.25, 1.75):                      # charleston (bruit bref)
        i = int(debut_s * fs)
        n = int(0.04 * fs)
        boucle[i: i + n] += 0.15 * rng.standard_normal(n) * np.exp(-np.arange(n) / (0.01 * fs))
    th = np.arange(2 * fs) / fs
    accord = np.where(th < 1.0, 1.0, 0.0)                         # 2 accords différents par boucle
    for f1, f2, f3, m in ((220, 277, 330, accord), (196, 247, 294, 1 - accord)):
        boucle += 0.08 * m * (np.sin(2 * np.pi * f1 * th) + np.sin(2 * np.pi * f2 * th)
                              + np.sin(2 * np.pi * f3 * th))
    instrumental = np.tile(boucle, int(duree // 2))
    # --- Voix : mélodie qui ne se répète jamais ---
    voix = voix_synthetique(fs, len(instrumental) / fs,
                            lambda t: 300 + 80 * np.sin(2 * np.pi * 0.37 * t)
                            + 40 * np.sin(2 * np.pi * 0.11 * t) + 6 * np.sin(2 * np.pi * 5 * t))
    return voix, instrumental, voix + instrumental, fs


def executer_demo_restauration(dossier, rappel: Optional[Rappel] = None) -> ResultatTraitement:
    """Génère le signal bruité, lance la VRAIE restauration, mesure le SNR contre le signal propre."""
    debut = time.perf_counter()
    dossier = Path(dossier)
    signaler(rappel, "Génération des signaux synthétiques…", 0.0)
    propre, bruite, fs = generer_demo_restauration()

    info: dict = {}
    restaure = restaurer_voix(bruite, fs, info=info,
                              rappel=lambda s, p: signaler(rappel, s, 0.1 + 0.6 * p))

    signaler(rappel, "Export des fichiers…", 0.75)
    f_bruite, f_restaure = dossier / "demo_voix_bruitee.wav", dossier / "demo_voix_restauree.wav"
    sauvegarder_audio(f_bruite, bruite, fs)
    sauvegarder_audio(f_restaure, restaure, fs)

    signaler(rappel, "Génération des graphiques…", 0.85)
    png = tracer_spectrogrammes({"Bruité": bruite, "Restauré": restaure, "Référence propre": propre}, fs,
                                dossier / "demo_restauration.png", "Démo restauration")
    avant, apres = snr_db(propre, bruite), snr_db(propre, restaure)
    signaler(rappel, "Terminé", 1.0)
    return ResultatTraitement(
        dossier=dossier,
        fichiers={"bruite": f_bruite, "restaure": f_restaure, "spectrogrammes": png},
        metriques=[
            Metrique("SNR réel avant", f"{avant:.1f} dB", "mesure",
                     "Calculé contre le signal propre synthétique, connu."),
            Metrique("SNR réel après", f"{apres:.1f} dB", "mesure"),
            Metrique("Amélioration mesurée", f"{apres - avant:+.1f} dB", "mesure"),
            Metrique("Fréquence de coupure du passe-haut", f"{info['coupure_hz']:.0f} Hz", "info"),
        ],
        notes=["Mesure valable pour ce signal synthétique ; sur un enregistrement réel, "
               "le résultat dépend du type de bruit."],
        duree_calcul_s=time.perf_counter() - debut)


def executer_demo_musique(methode: str, dossier, rappel: Optional[Rappel] = None) -> ResultatTraitement:
    """Génère voix + boucle, lance la VRAIE séparation, mesure les SNR contre les vraies sources."""
    debut = time.perf_counter()
    dossier = Path(dossier)
    if methode == "nmf":
        verifier_sklearn()
    signaler(rappel, "Génération des signaux synthétiques…", 0.0)
    voix, instrumental, melange, fs = generer_demo_musique()

    if methode == "nmf":
        signaler(rappel, "Séparation des sources (NMF)…", None)   # avancement inconnu
        voix_est, fond_est = separer_voix_fond(melange, fs, "nmf")
    else:
        signaler(rappel, "Séparation des sources (REPET-SIM)…", 0.1)
        voix_est, fond_est = separer_voix_fond(
            melange, fs, "repet",
            rappel_progression=lambda p: signaler(rappel, "Séparation des sources (REPET-SIM)…", 0.1 + 0.65 * p))

    signaler(rappel, "Export des fichiers…", 0.8)
    f_mel, f_voix, f_fond = (dossier / "demo_melange.wav", dossier / "demo_vocals.wav",
                             dossier / "demo_instrumental.wav")
    sauvegarder_audio(f_mel, melange, fs)
    sauvegarder_audio(f_voix, voix_est, fs)
    sauvegarder_audio(f_fond, fond_est, fs)

    signaler(rappel, "Génération des graphiques…", 0.9)
    png = tracer_spectrogrammes({"Mélange": melange, "Voix estimée": voix_est,
                                 "Instrumental estimé": fond_est}, fs,
                                dossier / "demo_musique.png", "Démo séparation")
    p_voix, p_fond = parts_energie(voix_est, fond_est)
    signaler(rappel, "Terminé", 1.0)
    return ResultatTraitement(
        dossier=dossier,
        fichiers={"melange": f_mel, "voix": f_voix, "instrumental": f_fond, "spectrogrammes": png},
        metriques=[
            Metrique("SNR voix : mélange brut", f"{snr_db(voix, melange):.1f} dB", "mesure",
                     "Contre la vraie voix synthétique, connue."),
            Metrique("SNR voix : après séparation", f"{snr_db(voix, voix_est):.1f} dB", "mesure"),
            Metrique("SNR instrumental : mélange brut", f"{snr_db(instrumental, melange):.1f} dB", "mesure"),
            Metrique("SNR instrumental : après séparation", f"{snr_db(instrumental, fond_est):.1f} dB", "mesure"),
            Metrique("Part d'énergie — voix estimée", f"{p_voix:.0f} %", "info"),
            Metrique("Part d'énergie — instrumental estimé", f"{p_fond:.0f} %", "info"),
        ],
        notes=["Cas favorable : l'instrumental est une boucle exactement répétée, ce qu'exploite REPET-SIM. "
               "Sur une vraie chanson, les résultats sont moins bons."],
        duree_calcul_s=time.perf_counter() - debut)
