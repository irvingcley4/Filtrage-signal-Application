"""Fenêtre principale : barre latérale + pages empilées (QStackedWidget).

Les pages sont créées UNE fois et conservées : revenir à l'accueil n'efface ni les fichiers
choisis ni les résultats, et les lecteurs ne sont pas recréés à chaque navigation."""
from __future__ import annotations

import os

from PySide6.QtWidgets import (QButtonGroup, QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QStackedWidget, QVBoxLayout, QWidget)

from gui.demo_page import PageDemo
from gui.home_page import DialogDependances, HomePage, TITRE_APPLICATION
from gui.music_page import PageMusique
from gui.podcast_page import PagePodcast
from gui.restoration_page import PageRestauration
from gui.result_widgets import demander_confirmation

DELAI_ARRET_MS = 3000        # attente de l'arrêt coopératif avant de proposer de forcer la fermeture


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(TITRE_APPLICATION)
        self.resize(1180, 820)
        self.setMinimumSize(860, 600)

        centre = QWidget()
        disposition = QHBoxLayout(centre)
        disposition.setContentsMargins(0, 0, 0, 0)
        disposition.setSpacing(0)
        self.setCentralWidget(centre)

        # --- pages (ordre = index de la pile) ---
        self._accueil = HomePage()
        self._pages = [PageMusique(), PageRestauration(), PagePodcast(), PageDemo()]
        self._pile = QStackedWidget()
        self._pile.addWidget(self._accueil)
        for page in self._pages:
            self._pile.addWidget(page)
            page.retour_accueil.connect(lambda: self.aller_a(0))
        self._accueil.ouvrir_page.connect(self.aller_a)

        # --- barre latérale ---
        barre = QFrame()
        barre.setObjectName("Sidebar")
        barre.setFixedWidth(230)
        v = QVBoxLayout(barre)
        v.setContentsMargins(14, 20, 14, 16)
        v.setSpacing(4)
        titre = QLabel("Audio Suite")
        titre.setObjectName("TitreSidebar")
        sous = QLabel("Séparation & restauration")
        sous.setObjectName("SousTitreSidebar")
        v.addWidget(titre)
        v.addWidget(sous)
        v.addSpacing(18)
        self._groupe = QButtonGroup(self)
        self._groupe.setExclusive(True)
        entrees = ["🏠  Accueil", "🎵  Séparer une musique", "🎙  Nettoyer une voix",
                   "📻  Podcast / Radio", "🧪  Démonstrations"]
        self._boutons = []
        for index, texte in enumerate(entrees):
            bouton = QPushButton(texte)
            bouton.setObjectName("NavBouton")
            bouton.setCheckable(True)
            bouton.clicked.connect(lambda _=False, i=index: self.aller_a(i))
            self._groupe.addButton(bouton, index)
            self._boutons.append(bouton)
            v.addWidget(bouton)
        v.addStretch(1)
        dependances = QPushButton("Dépendances")
        dependances.setObjectName("NavBouton")
        dependances.clicked.connect(self._ouvrir_dependances)
        v.addWidget(dependances)

        disposition.addWidget(barre)
        disposition.addWidget(self._pile, 1)
        self.aller_a(0)

    def aller_a(self, index: int) -> None:
        """Change de page sans rien réinitialiser."""
        self._pile.setCurrentIndex(index)
        self._boutons[index].setChecked(True)

    def _ouvrir_dependances(self) -> None:
        DialogDependances(self).exec()
        self._accueil.actualiser_dependances()

    def closeEvent(self, evenement) -> None:
        """Ne ferme pas brutalement pendant un calcul : demande confirmation, annule, attend la fin."""
        actives = [p for p in self._pages if p.tache_en_cours()]
        if actives:
            if not demander_confirmation(
                    self, "Traitement en cours",
                    "Un traitement est en cours. Voulez-vous l'annuler et quitter l'application ?",
                    oui="Annuler et quitter", non="Continuer le traitement"):
                evenement.ignore()
                return
            for page in actives:
                page.demander_annulation()
            for page in actives:
                if not page.attendre_fin(DELAI_ARRET_MS):
                    # Un bloc de calcul (ex. Demucs sur CPU) peut durer longtemps et n'est interruptible
                    # qu'entre deux blocs : on propose d'attendre ou de forcer la fermeture.
                    if demander_confirmation(
                            self, "Arrêt impossible pour le moment",
                            "Le calcul en cours ne peut être interrompu qu'à la fin de son étape actuelle. "
                            "Voulez-vous forcer la fermeture ? Le traitement en cours sera perdu.",
                            oui="Forcer la fermeture", non="Attendre"):
                        for p in self._pages:
                            p.liberer_ressources()
                        os._exit(0)             # seul moyen d'arrêter un calcul PyTorch en cours
                    evenement.ignore()
                    return
        for page in self._pages:
            page.liberer_ressources()
        evenement.accept()
