"""Exceptions de l'application.

Les messages des exceptions `AudioError` et `DependanceManquanteError` sont écrits
pour être affichés tels quels à l'utilisateur (en français).
"""


class AudioError(Exception):
    """Erreur liée à un fichier audio ou à un paramètre de traitement."""


class FfmpegIntrouvableError(AudioError):
    """FFmpeg est nécessaire pour lire ce format mais n'est pas installé."""


class FichierInvalideError(AudioError):
    """Fichier absent, vide, illisible ou sans piste audio."""


class AudioTropCourtError(AudioError):
    """Signal trop court pour que l'algorithme soit fiable."""


class DependanceManquanteError(Exception):
    """Une bibliothèque Python optionnelle (ex. scikit-learn) est absente."""


class TraitementAnnule(Exception):
    """Levée dans le thread de calcul quand l'utilisateur demande l'annulation."""
