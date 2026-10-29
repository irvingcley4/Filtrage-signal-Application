"""Traitements en arrière-plan : un worker exécute une fonction dans un QThread et
communique UNIQUEMENT par signaux Qt (les widgets ne sont jamais touchés depuis le thread de calcul)."""
from __future__ import annotations

import traceback
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, QThread, Signal, Slot

from core.erreurs import AudioError, DependanceManquanteError, TraitementAnnule


class TraitementWorker(QObject):
    """Exécute `fonction(*args, rappel=..., **kwargs)` ; vit dans le thread de calcul."""

    statut = Signal(str)            # message d'étape
    progression = Signal(float)     # 0..1 ; valeur négative = avancement inconnu (barre indéterminée)
    termine = Signal(object)        # résultat de la fonction
    erreur = Signal(str, str)       # message lisible, trace technique complète
    annule = Signal()
    fini = Signal()                 # toujours émis en dernier, quoi qu'il arrive

    def __init__(self, fonction: Callable[..., Any], *args, avec_rappel: bool = True, **kwargs):
        super().__init__()
        self._fonction = fonction
        self._args = args
        self._kwargs = kwargs
        self._avec_rappel = avec_rappel
        self._annulation_demandee = False

    def demander_annulation(self) -> None:
        """Appelée depuis le thread de l'interface ; prise en compte au prochain rappel."""
        self._annulation_demandee = True

    def _rappel(self, statut: str, progression: Optional[float] = None) -> None:
        """Appelé par l'algorithme (thread de calcul). Interrompt si l'annulation est demandée."""
        if self._annulation_demandee:
            raise TraitementAnnule()
        self.statut.emit(statut)
        self.progression.emit(-1.0 if progression is None else float(progression))

    @Slot()
    def executer(self) -> None:
        try:
            kwargs = dict(self._kwargs)
            if self._avec_rappel:
                kwargs["rappel"] = self._rappel
            resultat = self._fonction(*self._args, **kwargs)
        except TraitementAnnule:
            self.annule.emit()
        except (AudioError, DependanceManquanteError) as exc:
            self.erreur.emit(str(exc), traceback.format_exc())
        except MemoryError:
            self.erreur.emit("Mémoire insuffisante pour ce fichier. Essayez une fréquence de travail "
                             "plus basse ou un fichier plus court.", traceback.format_exc())
        except Exception:
            self.erreur.emit("Une erreur inattendue est survenue pendant le traitement. "
                             "Ouvrez « Détails techniques » pour plus d'informations.",
                             traceback.format_exc())
        else:
            self.termine.emit(resultat)
        finally:
            self.fini.emit()


class ExecuteurTache(QObject):
    """Gère un worker + son QThread et relaie ses signaux dans le thread de l'interface.
    Une seule tâche à la fois. Les pages se connectent à CES signaux."""

    statut = Signal(str)
    progression = Signal(float)
    termine = Signal(object)
    erreur = Signal(str, str)
    annule = Signal()
    fini = Signal()                 # émis une fois le thread réellement terminé et nettoyé

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._worker: Optional[TraitementWorker] = None
        self._thread: Optional[QThread] = None

    def en_cours(self) -> bool:
        return self._thread is not None

    def lancer(self, fonction: Callable[..., Any], *args, avec_rappel: bool = True, **kwargs) -> None:
        if self.en_cours():
            raise RuntimeError("Une tâche est déjà en cours.")
        worker = TraitementWorker(fonction, *args, avec_rappel=avec_rappel, **kwargs)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.executer)
        # Slots d'un QObject du thread principal : Qt met les appels en file (thread-safe).
        worker.statut.connect(self._relayer_statut)
        worker.progression.connect(self._relayer_progression)
        worker.termine.connect(self._relayer_termine)
        worker.erreur.connect(self._relayer_erreur)
        worker.annule.connect(self._relayer_annule)
        worker.fini.connect(thread.quit)
        thread.finished.connect(self._nettoyer)
        self._worker, self._thread = worker, thread
        thread.start()

    def demander_annulation(self) -> None:
        if self._worker is not None:
            self._worker.demander_annulation()

    def attendre_fin(self, delai_ms: int) -> bool:
        """Bloque (thread principal) jusqu'à la fin du thread ; False si délai dépassé."""
        if self._thread is None:
            return True
        return bool(self._thread.wait(delai_ms))

    @Slot(str)
    def _relayer_statut(self, texte: str) -> None:
        self.statut.emit(texte)

    @Slot(float)
    def _relayer_progression(self, valeur: float) -> None:
        self.progression.emit(valeur)

    @Slot(object)
    def _relayer_termine(self, resultat) -> None:
        self.termine.emit(resultat)

    @Slot(str, str)
    def _relayer_erreur(self, message: str, trace: str) -> None:
        self.erreur.emit(message, trace)

    @Slot()
    def _relayer_annule(self) -> None:
        self.annule.emit()

    @Slot()
    def _nettoyer(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self.fini.emit()
