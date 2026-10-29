"""MODE B : podcast / radio = segmentation parole / musique / mixte / silence,
puis isolation de la voix et du tapis sonore."""
import numpy as np
from scipy.ndimage import uniform_filter1d

from .dsp import EPS
from .erreurs import AudioTropCourtError
from .separation import separer_voix_fond

# Seuils de la segmentation (à ajuster selon le type d'émission)
SEUIL_SILENCE_DB = 40      # une fenêtre est "silence" si elle est 40 dB sous la plus forte
SEUIL_VARIATION = 0.5      # coefficient de variation de l'enveloppe : > seuil -> "ça parle"
SEUIL_PLANCHER_DB = -18    # écart pauses/pics : > seuil -> quelque chose joue sous la voix

# Couleurs uniques pour les graphiques, le tableau et la légende
COULEURS_SEGMENTS = {"parole": "#4c9be8", "musique": "#e8884c", "mixte": "#9a6ae0", "silence": "#cccccc"}
LIBELLES_SEGMENTS = {"parole": "Parole", "musique": "Musique", "mixte": "Mixte (parole + musique)",
                     "silence": "Silence"}


def _enveloppe_rms(x, fs):
    """Enveloppe d'énergie : une valeur RMS par trame de 25 ms, toutes les 10 ms.
    Calculée par paquets de trames : mêmes valeurs que le calcul d'un seul bloc,
    mais sans allouer plusieurs Go de mémoire pour un long enregistrement."""
    trame, hop = int(0.025 * fs), int(0.010 * fs)
    if len(x) < trame:
        raise AudioTropCourtError("Ce fichier est trop court pour analyser un podcast de façon fiable.")
    n = max(1, (len(x) - trame) // hop + 1)
    rms = np.empty(n)
    base = np.arange(trame)[None, :]
    pas = 4096
    for debut in range(0, n, pas):
        fin = min(debut + pas, n)
        idx = base + hop * np.arange(debut, fin)[:, None]
        rms[debut:fin] = np.sqrt((x[idx] ** 2).mean(axis=1))
    return rms


def segmenter_parole_musique(x, fs, duree_fenetre=1.0):
    """Étiquette chaque fenêtre de 1 s : 'silence', 'parole', 'musique' ou 'mixte'.
    Deux observations simples :
      - La parole est très IRRÉGULIÈRE (syllabes, pauses) : l'enveloppe d'énergie varie
        beaucoup  -> fort 'coefficient de variation' (écart-type / moyenne).
      - Dans de la parole PURE, les pauses sont presque silencieuses ; si pendant les
        pauses le niveau reste élevé, c'est qu'une musique tourne dessous -> 'mixte'.
    Retourne (liste d'étiquettes, enveloppe en dB, pas de l'enveloppe en s)."""
    rms = _enveloppe_rms(x, fs)
    n = len(rms)
    env_db = 20 * np.log10(rms + 1e-8)

    par_fenetre = int(duree_fenetre / 0.010)                  # trames par fenêtre
    niveau_max = env_db.max()
    etiquettes = []
    for debut in range(0, n, par_fenetre):
        e = rms[debut: debut + par_fenetre]
        niveau = 20 * np.log10(e.mean() + 1e-8)
        if niveau < niveau_max - SEUIL_SILENCE_DB:
            etiquettes.append("silence")
            continue
        variation = e.std() / (e.mean() + EPS)
        plancher_db = 20 * np.log10((np.percentile(e, 10) + EPS) / (np.percentile(e, 90) + EPS))
        if variation < SEUIL_VARIATION:
            etiquettes.append("musique")
        elif plancher_db > SEUIL_PLANCHER_DB:
            etiquettes.append("mixte")
        else:
            etiquettes.append("parole")
    return etiquettes, env_db, 0.010


def fusionner_segments(etiquettes, duree_fenetre=1.0):
    """Regroupe les fenêtres consécutives de même type : [(début_s, fin_s, type), ...]."""
    segments, debut = [], 0
    for i in range(1, len(etiquettes) + 1):
        if i == len(etiquettes) or etiquettes[i] != etiquettes[debut]:
            segments.append((debut * duree_fenetre, i * duree_fenetre, etiquettes[debut]))
            debut = i
    return segments


def segments_pour_affichage(etiquettes, duree_s):
    """Segments fusionnés, bornés à la durée réelle (la dernière fenêtre peut dépasser
    la fin du fichier à cause de l'arrondi ; une fenêtre vide est ignorée)."""
    resultat = []
    for debut, fin, type_ in fusionner_segments(etiquettes):
        if debut >= duree_s:
            continue
        resultat.append((debut, min(fin, duree_s), type_))
    return resultat


def isoler_voix_tapis(x, fs, rappel_progression=None):
    """Sépare voix et tapis sonore selon le type de chaque segment :
      parole  -> tout va dans la voix        musique -> tout va dans le fond
      mixte   -> séparation REPET-SIM         silence -> rien
    Les transitions sont adoucies (fondu de 0,3 s) pour éviter les clics.

    Optimisation sans effet sur le résultat : la séparation REPET-SIM (coûteuse) n'est
    lancée que s'il existe au moins un segment « mixte ». Sinon son résultat serait
    multiplié par un poids nul partout."""
    etiquettes, env_db, pas = segmenter_parole_musique(x, fs)
    if "mixte" in etiquettes:
        voix_sep, fond_sep = separer_voix_fond(x, fs, "repet", rappel_progression=rappel_progression)
    else:
        voix_sep = fond_sep = np.zeros(len(x))

    def courbe(type_cherche):
        """Courbe 0/1 au rythme des échantillons pour un type de segment, puis adoucie."""
        c = np.zeros(len(x))
        for i, e in enumerate(etiquettes):
            if e == type_cherche:
                c[int(i * fs): int((i + 1) * fs)] = 1.0
        return uniform_filter1d(c, size=int(0.3 * fs))

    w_parole, w_musique, w_mixte = courbe("parole"), courbe("musique"), courbe("mixte")
    voix = w_parole * x + w_mixte * voix_sep
    fond = w_musique * x + w_mixte * fond_sep
    return voix, fond, etiquettes, env_db, pas


# ======================================================================================
# Variante Demucs : les passages « mixtes » sont séparés par Demucs
# ======================================================================================
ECART_FUSION_S = 2.0   # deux passages mixtes séparés par moins de 2 s sont regroupés
CONTEXTE_S = 2.0       # contexte audio ajouté de part et d'autre d'un passage mixte pour Demucs
FONDU_S = 0.25         # durée des fondus enchaînés aux frontières (réduite si les segments sont courts)


def regrouper_mixtes(etiquettes, ecart_fusion_s=ECART_FUSION_S, duree_fenetre=1.0):
    """Regroupe les passages « mixte » proches : les fenêtres comprises entre deux passages mixtes
    séparés de moins de `ecart_fusion_s` deviennent « mixte » aussi. Demucs traite alors un seul passage
    continu au lieu de plusieurs fragments (moins de ruptures de timbre)."""
    etiq = list(etiquettes)
    indices = [i for i, e in enumerate(etiquettes) if e == "mixte"]
    for i, j in zip(indices, indices[1:]):
        if 0 < (j - i - 1) * duree_fenetre <= ecart_fusion_s:
            for m in range(i + 1, j):
                etiq[m] = "mixte"
    return etiq


def bornes_en_echantillons(segments, fs, n):
    """Convertit des segments (début_s, fin_s, type) en bornes entières contiguës (a, b) couvrant
    exactement [0, n) : ni trou, ni doublon, ni décalage. Les segments qui commencent après la fin
    du signal (fenêtre vide due à l'arrondi) sont ignorés. Retourne (bornes, types)."""
    utiles = [s for s in segments if int(round(s[0] * fs)) < n]
    bornes = []
    for i, (_, fin, _) in enumerate(utiles):
        a = 0 if i == 0 else bornes[-1][1]
        b = n if i == len(utiles) - 1 else min(n, int(round(fin * fs)))
        bornes.append((a, b))
    return bornes, [s[2] for s in utiles]


def calculer_ponderations(bornes, fs, n, fondu_s=FONDU_S):
    """Poids de chaque segment, avec fondus enchaînés centrés sur chaque frontière.
    La durée du fondu est min(fondu_s, moitié du plus court des deux segments voisins) : un passage
    très court n'est pas dégradé par un fondu trop long. Les poids de tous les segments somment à 1
    en chaque échantillon (aucun trou, aucun doublon). Retourne [(i0, i1, poids), ...]."""
    k = len(bornes)
    demi = []
    for i in range(k - 1):
        d_g = (bornes[i][1] - bornes[i][0]) / fs
        d_d = (bornes[i + 1][1] - bornes[i + 1][0]) / fs
        demi.append(int(round(min(fondu_s, 0.5 * min(d_g, d_d)) * fs / 2)))
    resultat = []
    for i, (a, b) in enumerate(bornes):
        h_g = demi[i - 1] if i > 0 else 0
        h_d = demi[i] if i < k - 1 else 0
        i0, i1 = max(0, a - h_g), min(n, b + h_d)
        t = np.arange(i0, i1)
        w = np.ones(i1 - i0, dtype=np.float64)
        if h_g > 0:
            w *= np.clip((t - (a - h_g) + 0.5) / (2 * h_g), 0.0, 1.0)
        if h_d > 0:
            w *= np.clip(((b + h_d) - t - 0.5) / (2 * h_d), 0.0, 1.0)
        resultat.append((i0, i1, w.astype(np.float32)))
    return resultat


def assembler_pistes(x, fs, bornes, types, sorties_mixtes, fondu_s=FONDU_S):
    """Reconstruit les pistes voix et fond (tableaux (C, n)) par fondus enchaînés :
      parole  -> signal ORIGINAL dans la voix        musique -> signal ORIGINAL dans le fond
      mixte   -> sorties de Demucs (vocals / instrumental)   silence -> rien
    sorties_mixtes : {indice_du_segment: (début_du_bloc_en_échantillons, voix, fond)} ; le bloc doit
    couvrir le segment avec une marge au moins égale à la demi-durée des fondus."""
    n = x.shape[1]
    voix = np.zeros(x.shape, dtype=np.float32)
    fond = np.zeros(x.shape, dtype=np.float32)
    for k, ((i0, i1, w), type_) in enumerate(zip(calculer_ponderations(bornes, fs, n, fondu_s), types)):
        if type_ == "parole":
            voix[:, i0:i1] += w * x[:, i0:i1]
        elif type_ == "musique":
            fond[:, i0:i1] += w * x[:, i0:i1]
        elif type_ == "mixte":
            c0, v, f = sorties_mixtes[k]
            if i0 < c0 or i1 - c0 > v.shape[1]:
                raise ValueError("Le bloc traité par Demucs ne couvre pas le passage mixte.")
            voix[:, i0:i1] += w * v[:, i0 - c0:i1 - c0]
            fond[:, i0:i1] += w * f[:, i0 - c0:i1 - c0]
    return voix, fond


def isoler_voix_tapis_demucs(x, fs, separateur, rappel=None, infos=None):
    """Version Demucs de `isoler_voix_tapis`. x : signal STÉRÉO (2, n) ; `separateur` : SeparateurDemucs
    (ou objet équivalent avec .charger() et .separer(audio, rappel)).

    1. Segmentation parole / musique / mixte / silence (inchangée, sur le mono).
    2. Enregistrement sans musique ni passage mixte : la voix est le signal ORIGINAL, sans séparation.
    3. Passages mixtes proches regroupés ; chacun est traité par Demucs avec du contexte autour.
    4. Pistes reconstruites par fondus enchaînés (voir assembler_pistes).
    Retourne (voix, fond, étiquettes, env_db, pas), comme isoler_voix_tapis."""
    from .resultats import signaler

    n = x.shape[1]
    etiquettes, env_db, pas = segmenter_parole_musique(x.mean(axis=0), fs)
    if not ({"musique", "mixte"} & set(etiquettes)):
        if infos is not None:
            infos.update(mode="parole_seule", regions=0, duree_demucs_s=0.0)
        return x.astype(np.float32).copy(), np.zeros(x.shape, dtype=np.float32), etiquettes, env_db, pas

    segments = fusionner_segments(regrouper_mixtes(etiquettes))
    bornes, types = bornes_en_echantillons(segments, fs, n)
    regions = [k for k, t in enumerate(types) if t == "mixte"]
    sorties = {}
    contexte = int(CONTEXTE_S * fs)
    for rang, k in enumerate(regions):
        a, b = bornes[k]
        s0, s1 = max(0, a - contexte), min(n, b + contexte)

        def rappel_region(statut, progression, rang=rang):
            fraction = (rang + (progression or 0.0)) / len(regions)
            signaler(rappel, f"Passage mixte {rang + 1}/{len(regions)} : {statut}", fraction)

        signaler(rappel, f"Séparation du passage mixte {rang + 1}/{len(regions)}…", rang / len(regions))
        voix_c, fond_c = separateur.separer(x[:, s0:s1], rappel=rappel_region)
        sorties[k] = (s0, voix_c, fond_c)
    voix, fond = assembler_pistes(x, fs, bornes, types, sorties)
    if infos is not None:
        infos.update(mode="segments", regions=len(regions),
                     duree_demucs_s=sum(bornes[k][1] - bornes[k][0] for k in regions) / fs)
    return voix, fond, etiquettes, env_db, pas
