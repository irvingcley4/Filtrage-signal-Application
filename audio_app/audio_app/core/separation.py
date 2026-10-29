"""MODE A : séparation voix / instrumental (REPET-SIM et NMF)."""
import numpy as np
from scipy.stats import kurtosis

from .dsp import EPS, calculer_stft, inverser_stft, masque_doux
from .erreurs import AudioTropCourtError, DependanceManquanteError


def verifier_sklearn():
    """Lève DependanceManquanteError si scikit-learn (nécessaire à NMF) est absent."""
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise DependanceManquanteError(
            "La dépendance scikit-learn est nécessaire pour utiliser NMF. "
            "Installez-la avec : python -m pip install scikit-learn") from exc


def fond_par_voisins_similaires(mag, fs, hop, k=20, taille_bloc=128, rappel_progression=None):
    """Méthode REPET-SIM : estime la partie RÉPÉTITIVE de la musique (le fond).
    Idée clé : un instrumental (batterie, basse, boucles) se répète ; la voix, non.
      1. Pour chaque trame, on cherche les k trames les plus SIMILAIRES (similarité cosinus).
      2. On prend la MÉDIANE de leurs spectres : ce qui revient partout est conservé,
         ce qui est unique à cet instant (la voix) est éliminé par la médiane.
    On exclut les trames voisines (~0,4 s) pour que la voix tenue ne se 'reconnaisse' pas elle-même.

    rappel_progression : fonction optionnelle f(fraction 0..1). L'avancement est RÉEL :
    c'est la proportion de blocs de trames déjà traités."""
    n_freq, T = mag.shape
    ecart = max(1, int(0.4 * fs / hop))
    if T < 2 * ecart + 5:
        raise AudioTropCourtError(
            "Ce fichier est trop court pour réaliser une séparation fiable.")
    k = int(max(1, min(k, T - (2 * ecart + 1) - 1)))

    X = (mag / (np.linalg.norm(mag, axis=0, keepdims=True) + EPS)).astype(np.float32)
    fond = np.empty_like(mag)
    for debut in range(0, T, taille_bloc):          # travail par blocs : économise la mémoire
        fin = min(debut + taille_bloc, T)
        S = X[:, debut:fin].T @ X                   # similarité (bloc x toutes les trames)
        for i, col in enumerate(range(debut, fin)):
            S[i, max(0, col - ecart): col + ecart + 1] = -np.inf   # on s'exclut soi-même
        voisins = np.argpartition(-S, k, axis=1)[:, :k]            # indices des k + similaires
        fond[:, debut:fin] = np.median(mag[:, voisins], axis=2)    # médiane de leurs spectres
        if rappel_progression is not None:
            rappel_progression(fin / T)
    return fond


def masques_repet(mag, fs, hop, marge_fond=2.0, marge_voix=4.0, rappel_progression=None):
    """Masques voix / fond avec REPET-SIM.
    fond  = partie répétitive (borné par le mélange : on ne peut pas avoir plus que ce qui existe)
    voix  = ce qui reste (mélange - fond)
    Les 'marges' rendent les masques plus sélectifs : moins de fuite de la musique dans la voix,
    au prix d'un peu de voix perdue. À régler à l'oreille."""
    fond = np.minimum(
        fond_par_voisins_similaires(mag, fs, hop, rappel_progression=rappel_progression), mag)
    residu = mag - fond
    masque_fond = masque_doux(fond, marge_fond * residu)
    masque_voix = masque_doux(residu, marge_voix * fond)
    return masque_voix, masque_fond


def masques_nmf(mag, fs, n_fft, n_composantes=24):
    """Séparation par NMF (Non-negative Matrix Factorization).
    On approxime le spectrogramme (positif) par un produit de deux matrices positives :
        mag.T  ~  W @ H
        H : n_composantes 'briques spectrales' (ex. : un kick, une note de basse, une voyelle...)
        W : à quel moment chaque brique est active
    Ensuite on range chaque brique dans 'voix' ou 'instrumental' avec une HEURISTIQUE :
      - part d'énergie dans la bande de la voix (200-4000 Hz)  -> plutôt voix
      - brique très 'explosive' dans le temps (kurtosis forte)  -> percussion -> instrumental
      - brique très grave (centroïde < 250 Hz)                  -> basse/kick  -> instrumental
    Plus fragile que REPET-SIM, mais montre bien le principe de factorisation.
    scikit-learn ne fournit pas l'avancement de l'ajustement : l'appelant doit afficher
    une progression indéterminée pendant cette étape."""
    verifier_sklearn()
    from sklearn.decomposition import NMF

    modele = NMF(n_components=n_composantes, init="nndsvda", solver="mu",
                 beta_loss="kullback-leibler", max_iter=150, random_state=0)
    W = modele.fit_transform(mag.T + EPS)       # activations (temps x composantes)
    H = modele.components_                      # briques spectrales (composantes x fréquences)

    freqs = np.linspace(0, fs / 2, mag.shape[0])
    bande_voix = (freqs >= 200) & (freqs <= 4000)
    scores = np.zeros(n_composantes)
    for c in range(n_composantes):
        p = H[c] ** 2
        ratio_bande = p[bande_voix].sum() / (p.sum() + EPS)
        centroide = (freqs * p).sum() / (p.sum() + EPS)
        explosif = np.clip(kurtosis(W[:, c]) / 20.0, 0.0, 1.0)
        scores[c] = ratio_bande - 0.5 * explosif - (0.5 if centroide < 250 else 0.0)
    est_voix = scores >= np.median(scores)      # la moitié "la plus vocale"

    voix = (W[:, est_voix] @ H[est_voix]).T
    fond = (W[:, ~est_voix] @ H[~est_voix]).T
    masque_voix = masque_doux(voix, fond, puissance=1)
    return masque_voix, 1.0 - masque_voix


def separer_voix_fond(x, fs, methode="repet", n_fft=2048, hop=1024, rappel_progression=None):
    """Sépare x en (voix, instrumental) avec un masque appliqué sur la STFT.
    rappel_progression : f(fraction 0..1), appelée seulement pour REPET-SIM (avancement réel)."""
    Z = calculer_stft(x, fs, n_fft, hop)
    mag = np.abs(Z).astype(np.float32)
    if methode == "nmf":
        m_voix, m_fond = masques_nmf(mag, fs, n_fft)
    else:
        m_voix, m_fond = masques_repet(mag, fs, hop, rappel_progression=rappel_progression)
    voix = inverser_stft(Z * m_voix, fs, n_fft, hop, len(x))
    fond = inverser_stft(Z * m_fond, fs, n_fft, hop, len(x))
    return voix, fond
