"""Traitements complets des trois modes, appelés par l'interface (via un worker).

Chaque fonction : charge le fichier, appelle les algorithmes existants, écrit les
fichiers dans `dossier`, calcule les métriques et renvoie un ResultatTraitement.
Le paramètre `rappel(statut, progression)` sert à informer l'interface ; il peut lever
TraitementAnnule pour interrompre proprement le calcul.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from .audio_io import (charger_audio, charger_audio_stereo, reechantillonner, sauvegarder_audio,
                       verifier_duree_minimale, verifier_wav_exporte)
from .demucs_separator import (FREQUENCE_DEMUCS, MODELE_PAR_DEFAUT, SeparateurDemucs,
                               verifier_demucs)
from .erreurs import AudioError
from .metrics import parts_energie, snr_estime
from .plots import tracer_segmentation, tracer_spectrogrammes
from .podcast import (LIBELLES_SEGMENTS, isoler_voix_tapis, isoler_voix_tapis_demucs,
                      segments_pour_affichage)
from .restoration import restaurer_voix
from .resultats import Metrique, Rappel, ResultatTraitement, signaler
from .separation import separer_voix_fond, verifier_sklearn

DUREE_MIN_SEPARATION_S = 5.0
DUREE_MIN_RESTAURATION_S = 2.0


def traiter_restauration(chemin, alpha: float, dossier, rappel: Optional[Rappel] = None) -> ResultatTraitement:
    """MODE C : passe-haut dynamique + soustraction spectrale (paramètre alpha)."""
    debut = time.perf_counter()
    dossier = Path(dossier)
    if not 1.0 <= alpha <= 3.0:
        raise AudioError("Le paramètre alpha doit être compris entre 1,0 et 3,0.")

    signaler(rappel, "Chargement du fichier audio…", 0.0)
    x, fs = charger_audio(chemin)
    verifier_duree_minimale(x, fs, DUREE_MIN_RESTAURATION_S,
                            "Ce fichier est trop court pour nettoyer la voix de façon fiable.")

    info: dict = {}
    propre = restaurer_voix(x, fs, alpha=alpha, info=info,
                            rappel=lambda s, p: signaler(rappel, s, 0.1 + 0.6 * p))

    signaler(rappel, "Export des fichiers…", 0.75)
    fichier_wav = dossier / "voix_restauree.wav"
    sauvegarder_audio(fichier_wav, propre, fs)

    signaler(rappel, "Génération des graphiques…", 0.85)
    png = tracer_spectrogrammes({"Avant (bruité)": x, "Après (restauré)": propre}, fs,
                                dossier / "spectrogrammes_restauration.png",
                                "Mode C - restauration vocale")

    avant, apres = snr_estime(x, fs), snr_estime(propre, fs)
    signaler(rappel, "Terminé", 1.0)
    return ResultatTraitement(
        dossier=dossier,
        fichiers={"original": Path(chemin), "restaure": fichier_wav, "spectrogrammes": png},
        metriques=[
            Metrique("Fréquence de coupure du passe-haut", f"{info['coupure_hz']:.0f} Hz", "info",
                     "Calculée automatiquement d'après la part d'énergie sous 150 Hz."),
            Metrique("Force de la soustraction (alpha)", f"{alpha:.1f}", "info"),
            Metrique("SNR estimé avant", f"{avant:.1f} dB", "estime",
                     "Estimation d'après les trames les plus faibles et les plus fortes."),
            Metrique("SNR estimé après", f"{apres:.1f} dB", "estime"),
            Metrique("Amélioration estimée", f"{apres - avant:+.1f} dB", "estime"),
        ],
        notes=["Sans signal propre de référence, le SNR n'est qu'ESTIMÉ : il sert à comparer "
               "avant et après, pas à donner une mesure exacte.",
               "Le passe-haut coupe aussi le bas des voix très graves (< ~120 Hz) : "
               "écoutez le résultat."],
        duree_calcul_s=time.perf_counter() - debut)


def _exporter_paire(chemin_a, piste_a, chemin_b, piste_b, fs) -> float:
    """Exporte deux pistes stéréo (2, n) en WAV 16 bits avec le MÊME gain (évite l'écrêtage sans modifier
    l'équilibre entre les deux), puis relit les en-têtes pour vérifier durée, fréquence et canaux.
    Retourne le gain appliqué (1.0 = aucun changement)."""
    pic = max(float(np.max(np.abs(piste_a))), float(np.max(np.abs(piste_b))))
    gain = 0.98 / pic if pic > 0.98 else 1.0
    n = piste_a.shape[1]
    for chemin, piste in ((chemin_a, piste_a), (chemin_b, piste_b)):
        sauvegarder_audio(chemin, (piste * gain).T, fs)
        verifier_wav_exporte(chemin, n, fs, 2)
    return gain


def _mono(audio):
    return audio.mean(axis=0)


def _traiter_musique_demucs(chemin, modele, dossier, rappel, debut) -> ResultatTraitement:
    """MODE A avec Demucs : vocals + instrumental (= somme des autres stems du modèle)."""
    verifier_demucs()                                   # échoue tôt, avant de décoder le fichier
    signaler(rappel, "Préparation du fichier audio…", 0.0)
    audio, fs = charger_audio_stereo(chemin, FREQUENCE_DEMUCS)
    verifier_duree_minimale(audio[0], fs, DUREE_MIN_SEPARATION_S,
                            "Ce fichier est trop court pour réaliser une séparation fiable.")

    separateur = SeparateurDemucs(modele, rappel=lambda s, p: signaler(rappel, s, None if p is None else 0.05 + 0.8 * p))
    try:
        signaler(rappel, "Chargement du modèle de séparation…", None)
        separateur.charger()
        if separateur.samplerate != fs:                 # modèle à une autre fréquence que 44100 Hz
            audio, fs = reechantillonner(audio, fs, separateur.samplerate).astype(np.float32), separateur.samplerate
        voix, instrumental = separateur.separer(audio)
        infos = separateur.infos()
        avertissements = list(separateur.avertissements)
    finally:
        separateur.liberer()

    signaler(rappel, "Export des résultats…", 0.9)
    f_voix, f_fond = dossier / "vocals.wav", dossier / "instrumental.wav"
    gain = _exporter_paire(f_voix, voix, f_fond, instrumental, fs)

    signaler(rappel, "Génération des graphiques…", 0.95)
    png = tracer_spectrogrammes({"Mélange": _mono(audio), "Voix isolée": _mono(voix),
                                 "Instrumental": _mono(instrumental)}, fs,
                                dossier / "spectrogrammes_musique.png", "Mode A - séparation par Demucs")
    p_voix, p_fond = parts_energie(voix, instrumental)
    duree = time.perf_counter() - debut
    notes = ["La qualité d'une séparation sur une vraie chanson se juge à l'écoute : sans stems de "
             "référence, aucun SNR de séparation n'est calculable.",
             "Sorties en stéréo, 44100 Hz, WAV 16 bits. L'instrumental est la somme des sources autres que "
             "la voix (batterie + basse + autres) fournies par le modèle."]
    if gain < 1.0:
        notes.append(f"Niveau réduit de {-20 * np.log10(gain):.1f} dB (identique pour les deux fichiers) pour éviter l'écrêtage.")
    if infos["device"] == "cpu":
        notes.append("Calcul effectué sur le processeur : un GPU NVIDIA (CUDA) serait nettement plus rapide.")
    notes.extend(avertissements)
    signaler(rappel, "Terminé", 1.0)
    return ResultatTraitement(
        dossier=dossier,
        fichiers={"original": Path(chemin), "voix": f_voix, "instrumental": f_fond, "spectrogrammes": png},
        metriques=[
            Metrique("Méthode", "Demucs", "info"),
            Metrique("Modèle", infos["modele"], "info"),
            Metrique("Périphérique de calcul", infos["peripherique"], "info"),
            Metrique("Temps de traitement", f"{duree:.0f} s", "info"),
            Metrique("Part d'énergie — voix isolée", f"{p_voix:.0f} %", "info",
                     "Répartition de l'énergie entre les deux signaux estimés ; ce n'est pas une note de qualité."),
            Metrique("Part d'énergie — instrumental", f"{p_fond:.0f} %", "info"),
        ],
        notes=notes, duree_calcul_s=duree)


def traiter_musique(chemin, methode: str, sr: int, dossier, rappel: Optional[Rappel] = None,
                    modele: str = MODELE_PAR_DEFAUT) -> ResultatTraitement:
    """MODE A : séparation voix / instrumental par Demucs (recommandé), REPET-SIM ou NMF.
    `sr` ne concerne que REPET-SIM et NMF ; Demucs travaille à la fréquence de son modèle (44100 Hz)."""
    debut = time.perf_counter()
    dossier = Path(dossier)
    if methode not in ("demucs", "repet", "nmf"):
        raise AudioError("Méthode de séparation inconnue.")
    if methode == "demucs":
        return _traiter_musique_demucs(chemin, modele, dossier, rappel, debut)
    if methode == "nmf":
        verifier_sklearn()

    signaler(rappel, "Chargement du fichier audio…", 0.0)
    x, fs = charger_audio(chemin, sr_cible=sr)
    verifier_duree_minimale(x, fs, DUREE_MIN_SEPARATION_S,
                            "Ce fichier est trop court pour réaliser une séparation fiable.")

    signaler(rappel, "Préparation du signal…", 0.1)
    if methode == "nmf":
        # scikit-learn ne donne pas l'avancement : progression indéterminée (None)
        signaler(rappel, "Séparation des sources (NMF)…", None)
        voix, fond = separer_voix_fond(x, fs, "nmf")
    else:
        signaler(rappel, "Séparation des sources (REPET-SIM)…", 0.15)
        voix, fond = separer_voix_fond(
            x, fs, "repet",
            rappel_progression=lambda p: signaler(rappel, "Séparation des sources (REPET-SIM)…", 0.15 + 0.6 * p))

    signaler(rappel, "Export des fichiers…", 0.8)
    f_voix, f_fond = dossier / "vocals.wav", dossier / "instrumental.wav"
    sauvegarder_audio(f_voix, voix, fs)
    sauvegarder_audio(f_fond, fond, fs)

    signaler(rappel, "Génération des graphiques…", 0.9)
    png = tracer_spectrogrammes({"Mélange": x, "Voix isolée": voix, "Instrumental": fond}, fs,
                                dossier / "spectrogrammes_musique.png", "Mode A - séparation de sources")

    p_voix, p_fond = parts_energie(voix, fond)
    notes = [f"Traitement effectué à {fs} Hz (contenu utile jusqu'à {fs // 2} Hz).",
             "Sans enregistrements séparés de référence, aucun SNR de séparation n'est calculable : "
             "jugez le résultat à l'écoute."]
    if methode == "repet":
        notes.append("REPET-SIM suppose que l'instrumental se répète : des chœurs, des voix doublées "
                     "ou des instruments très variables peuvent subsister dans la voix isolée.")
    else:
        notes.append("NMF est expérimentale : le classement des composantes voix/instrumental "
                     "est une heuristique, moins fiable que REPET-SIM.")
    signaler(rappel, "Terminé", 1.0)
    return ResultatTraitement(
        dossier=dossier,
        fichiers={"original": Path(chemin), "voix": f_voix, "instrumental": f_fond, "spectrogrammes": png},
        metriques=[
            Metrique("Part d'énergie — voix isolée", f"{p_voix:.0f} %", "info",
                     "Répartition de l'énergie entre les deux signaux estimés ; ce n'est pas une note de qualité."),
            Metrique("Part d'énergie — instrumental", f"{p_fond:.0f} %", "info"),
            Metrique("Méthode", "REPET-SIM" if methode == "repet" else "NMF", "info"),
        ],
        notes=notes, duree_calcul_s=time.perf_counter() - debut)


def _traiter_podcast_demucs(chemin, modele, dossier, rappel, debut) -> ResultatTraitement:
    """MODE B avec Demucs : parole / musique conservées telles quelles, passages mixtes séparés par Demucs."""
    verifier_demucs()
    signaler(rappel, "Préparation du fichier audio…", 0.0)
    audio, fs = charger_audio_stereo(chemin, FREQUENCE_DEMUCS)
    verifier_duree_minimale(audio[0], fs, DUREE_MIN_SEPARATION_S,
                            "Ce fichier est trop court pour analyser un podcast de façon fiable.")
    duree = audio.shape[1] / fs

    separateur = SeparateurDemucs(modele)
    details: dict = {}
    try:
        signaler(rappel, "Détection des segments…", 0.05)
        voix, fond, etiquettes, env_db, pas = isoler_voix_tapis_demucs(
            audio, fs, separateur,
            rappel=lambda s, p: signaler(rappel, s, None if p is None else 0.1 + 0.7 * p),
            infos=details)
        infos = separateur.infos() if separateur.modele is not None else None
        avertissements = list(separateur.avertissements)
    finally:
        separateur.liberer()

    signaler(rappel, "Export des résultats…", 0.85)
    f_voix, f_fond = dossier / "voix_podcast_claire.wav", dossier / "musique_de_fond.wav"
    gain = _exporter_paire(f_voix, voix, f_fond, fond, fs)

    signaler(rappel, "Génération des graphiques…", 0.92)
    png_seg = tracer_segmentation(env_db, pas, etiquettes, dossier / "segmentation_podcast.png")
    png_spec = tracer_spectrogrammes({"Émission": _mono(audio), "Voix claire": _mono(voix),
                                      "Musique de fond": _mono(fond)}, fs,
                                     dossier / "spectrogrammes_podcast.png", "Mode B - podcast / radio (Demucs)")

    segments = segments_pour_affichage(etiquettes, duree)
    repartition = []
    for type_ in ("parole", "musique", "mixte", "silence"):
        part = sum(f - d for d, f, t in segments if t == type_)
        if part > 0:
            repartition.append(f"{LIBELLES_SEGMENTS[type_].split(' (')[0]} {100 * part / duree:.0f} %")
    total = time.perf_counter() - debut
    notes = ["Segments de parole : signal original conservé dans la voix ; segments de musique : signal original "
             "conservé dans le fond. Demucs ne traite que les passages mixtes (fondus enchaînés de 250 ms au plus "
             "aux frontières, passages mixtes proches regroupés).",
             "Limite : Demucs est entraîné sur de la musique ; sa sortie « vocals » peut altérer une voix parlée "
             "(timbre, souffle) ou laisser passer de la musique. Écoutez les passages mixtes pour juger.",
             "La segmentation reste heuristique (seuils fixes, fenêtres d'une seconde). Sorties en stéréo, 44100 Hz."]
    if details.get("mode") == "parole_seule":
        notes.append("Aucune musique détectée : la voix est le signal original, sans séparation ; "
                     "la piste « musique de fond » est silencieuse.")
    elif details.get("regions", 0) == 0:
        notes.append("Aucun passage mixte détecté : Demucs n'a pas été nécessaire (signal original conservé).")
    if gain < 1.0:
        notes.append(f"Niveau réduit de {-20 * np.log10(gain):.1f} dB (identique pour les deux fichiers) pour éviter l'écrêtage.")
    notes.extend(avertissements)
    metriques = [
        Metrique("Durée analysée", f"{int(duree) // 60:d} min {int(duree) % 60:02d} s", "info"),
        Metrique("Segments détectés", str(len(segments)), "info"),
        Metrique("Répartition", " · ".join(repartition) or "—", "info"),
        Metrique("Moteur (passages mixtes)", "Demucs" if details.get("regions", 0) else "non utilisé", "info"),
    ]
    if infos is not None:
        metriques += [Metrique("Modèle", infos["modele"], "info"),
                      Metrique("Périphérique de calcul", infos["peripherique"], "info"),
                      Metrique("Passages mixtes traités", f"{details['regions']} ({details['duree_demucs_s']:.0f} s)", "info")]
    metriques.append(Metrique("Temps de traitement", f"{total:.0f} s", "info"))
    signaler(rappel, "Terminé", 1.0)
    return ResultatTraitement(
        dossier=dossier,
        fichiers={"original": Path(chemin), "voix": f_voix, "fond": f_fond,
                  "segmentation": png_seg, "spectrogrammes": png_spec},
        metriques=metriques, segments=segments, notes=notes, duree_calcul_s=total)


def traiter_podcast(chemin, sr: int, dossier, rappel: Optional[Rappel] = None,
                    moteur: str = "demucs", modele: str = MODELE_PAR_DEFAUT) -> ResultatTraitement:
    """MODE B : segmentation parole/musique/mixte/silence + isolation voix / tapis sonore.
    moteur = « demucs » (recommandé) ou « repet » pour les passages mixtes ; `sr` ne sert qu'à REPET-SIM."""
    debut = time.perf_counter()
    dossier = Path(dossier)
    if moteur not in ("demucs", "repet"):
        raise AudioError("Moteur de séparation inconnu.")
    if moteur == "demucs":
        return _traiter_podcast_demucs(chemin, modele, dossier, rappel, debut)

    signaler(rappel, "Chargement du fichier audio…", 0.0)
    x, fs = charger_audio(chemin, sr_cible=sr)
    verifier_duree_minimale(x, fs, DUREE_MIN_SEPARATION_S,
                            "Ce fichier est trop court pour analyser un podcast de façon fiable.")
    duree = len(x) / fs

    signaler(rappel, "Détection des segments et séparation…", 0.1)
    voix, fond, etiquettes, env_db, pas = isoler_voix_tapis(
        x, fs, rappel_progression=lambda p: signaler(rappel, "Séparation des passages mixtes…", 0.15 + 0.6 * p))

    signaler(rappel, "Export des fichiers…", 0.8)
    f_voix, f_fond = dossier / "voix_podcast_claire.wav", dossier / "musique_de_fond.wav"
    sauvegarder_audio(f_voix, voix, fs)
    sauvegarder_audio(f_fond, fond, fs)

    signaler(rappel, "Génération des graphiques…", 0.88)
    png_seg = tracer_segmentation(env_db, pas, etiquettes, dossier / "segmentation_podcast.png")
    png_spec = tracer_spectrogrammes({"Émission": x, "Voix claire": voix, "Musique de fond": fond}, fs,
                                     dossier / "spectrogrammes_podcast.png", "Mode B - podcast / radio")

    segments = segments_pour_affichage(etiquettes, duree)
    repartition = []
    for type_ in ("parole", "musique", "mixte", "silence"):
        part = sum(f - d for d, f, t in segments if t == type_)
        if part > 0:
            repartition.append(f"{LIBELLES_SEGMENTS[type_].split(' (')[0]} {100 * part / duree:.0f} %")
    notes = ["La segmentation est heuristique (seuils fixes, fenêtres d'une seconde) : vérifiez-la "
             "à l'écoute. Une musique très forte sous la voix peut être classée « musique » plutôt que « mixte »."]
    if "mixte" not in etiquettes:
        notes.append("Aucun passage mixte détecté : la séparation REPET-SIM n'a pas été nécessaire "
                     "(les passages de parole sont copiés tels quels dans la voix).")
    signaler(rappel, "Terminé", 1.0)
    return ResultatTraitement(
        dossier=dossier,
        fichiers={"original": Path(chemin), "voix": f_voix, "fond": f_fond,
                  "segmentation": png_seg, "spectrogrammes": png_spec},
        metriques=[
            Metrique("Durée analysée", f"{int(duree) // 60:d} min {int(duree) % 60:02d} s", "info"),
            Metrique("Segments détectés", str(len(segments)), "info"),
            Metrique("Répartition", " · ".join(repartition) or "—", "info"),
        ],
        segments=segments, notes=notes, duree_calcul_s=time.perf_counter() - debut)
