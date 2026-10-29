"""Page d'accueil : quatre cartes de modes + zone des dépendances."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QPlainTextEdit,
                               QPushButton, QScrollArea, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget, QHeaderView, QAbstractItemView)

from core.environnement import Dependance, verifier_dependances
from gui.result_widgets import Carte, etiquette_secondaire
from gui.styles import ERREUR, SUCCES

TITRE_APPLICATION = "Audio Source Separation & Denoising Suite"
SOUS_TITRE = ("Séparez, restaurez et analysez vos fichiers audio : voix et instrumental, "
              "nettoyage d'enregistrements vocaux, podcasts et émissions de radio.")

# (badge, titre, description, index de page)
MODES = [
    ("🎵", "Séparer une musique", "Isoler les voix et l'instrumental d'une musique.", 1),
    ("🎙", "Nettoyer une voix",
     "Réduire le bruit, le souffle et les basses fréquences indésirables d'un enregistrement vocal.", 2),
    ("📻", "Podcast / Radio",
     "Analyser un enregistrement, détecter les segments et isoler la voix de la musique de fond.", 3),
    ("🧪", "Démonstrations",
     "Tester les algorithmes à partir de signaux synthétiques sans fournir de fichier audio.", 4),
]


class CarteMode(QFrame):
    def __init__(self, badge: str, titre: str, description: str, index: int, ouvrir, parent=None):
        super().__init__(parent)
        self.setObjectName("CarteMode")
        disposition = QVBoxLayout(self)
        disposition.setContentsMargins(22, 20, 22, 20)
        disposition.setSpacing(10)
        icone = QLabel(badge)
        icone.setObjectName("Badge")
        icone.setFixedSize(48, 48)
        icone.setAlignment(Qt.AlignmentFlag.AlignCenter)
        titre_lbl = QLabel(titre)
        titre_lbl.setObjectName("TitreMode")
        disposition.addWidget(icone)
        disposition.addWidget(titre_lbl)
        disposition.addWidget(etiquette_secondaire(description), 1)
        bouton = QPushButton("Commencer")
        bouton.setProperty("primaire", True)
        bouton.clicked.connect(lambda: ouvrir(index))
        disposition.addWidget(bouton)


class DialogDependances(QDialog):
    """Tableau de l'état des dépendances + commandes d'installation de celles qui manquent."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Vérification des dépendances")
        self.resize(720, 480)
        self._disposition = QVBoxLayout(self)
        self._tableau = QTableWidget(0, 4)
        self._tableau.setHorizontalHeaderLabels(["Composant", "Rôle", "État", "Version"])
        self._tableau.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._tableau.verticalHeader().setVisible(False)
        self._tableau.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._tableau.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._disposition.addWidget(self._tableau, 1)
        self._disposition.addWidget(QLabel("Commandes d'installation des éléments manquants :"))
        self._commandes = QPlainTextEdit()
        self._commandes.setReadOnly(True)
        self._commandes.setMaximumHeight(130)
        self._disposition.addWidget(self._commandes)
        boutons = QHBoxLayout()
        actualiser = QPushButton("Vérifier à nouveau")
        actualiser.clicked.connect(self.actualiser)
        fermer = QPushButton("Fermer")
        fermer.clicked.connect(self.accept)
        boutons.addWidget(actualiser)
        boutons.addStretch(1)
        boutons.addWidget(fermer)
        self._disposition.addLayout(boutons)
        self.actualiser()

    def actualiser(self) -> None:
        dependances = verifier_dependances()
        self._tableau.setRowCount(len(dependances))
        manquantes: list[Dependance] = []
        for i, d in enumerate(dependances):
            etat = "✓ disponible" if d.disponible else "✗ manquant"
            for j, texte in enumerate((d.nom, d.role, etat, d.version)):
                self._tableau.setItem(i, j, QTableWidgetItem(texte))
            if not d.disponible:
                manquantes.append(d)
        self._commandes.setPlainText(
            "\n".join(f"{d.nom} : {d.commande_installation}" for d in manquantes)
            or "Toutes les dépendances sont installées.")


class HomePage(QWidget):
    ouvrir_page = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("Page")
        racine = QVBoxLayout(self)
        racine.setContentsMargins(0, 0, 0, 0)
        defilement = QScrollArea()
        defilement.setWidgetResizable(True)
        contenu = QWidget()
        contenu.setObjectName("Contenu")
        colonne = QVBoxLayout(contenu)
        colonne.setContentsMargins(32, 28, 32, 28)
        colonne.setSpacing(18)
        defilement.setWidget(contenu)
        racine.addWidget(defilement)

        titre = QLabel(TITRE_APPLICATION)
        titre.setObjectName("TitreApp")
        titre.setWordWrap(True)
        colonne.addWidget(titre)
        colonne.addWidget(etiquette_secondaire(SOUS_TITRE))

        grille = QGridLayout()
        grille.setSpacing(16)
        for i, (badge, nom, description, index) in enumerate(MODES):
            grille.addWidget(CarteMode(badge, nom, description, index, self.ouvrir_page.emit), i // 2, i % 2)
        grille.setColumnStretch(0, 1)
        grille.setColumnStretch(1, 1)
        colonne.addLayout(grille)

        self._carte_dependances = Carte("Dépendances principales")
        self._ligne_etat = QLabel()
        self._ligne_etat.setTextFormat(Qt.TextFormat.RichText)
        self._ligne_etat.setWordWrap(True)
        self._carte_dependances.contenu.addWidget(self._ligne_etat)
        ligne = QHBoxLayout()
        verifier = QPushButton("Vérifier les dépendances")
        verifier.clicked.connect(self.ouvrir_dialogue_dependances)
        ligne.addWidget(verifier)
        ligne.addStretch(1)
        self._carte_dependances.contenu.addLayout(ligne)
        colonne.addWidget(self._carte_dependances)
        colonne.addStretch(1)
        self.actualiser_dependances()

    def actualiser_dependances(self) -> None:
        morceaux = []
        for d in verifier_dependances():
            if d.nom == "PySide6":
                continue                       # forcément présent : l'application tourne
            couleur, signe = (SUCCES, "✓") if d.disponible else (ERREUR, "✗")
            morceaux.append(f'<span style="color:{couleur}; font-weight:600;">{signe}</span> {d.nom}')
        self._ligne_etat.setText("&nbsp;&nbsp;&nbsp;&nbsp;".join(morceaux))

    def ouvrir_dialogue_dependances(self) -> None:
        DialogDependances(self).exec()
        self.actualiser_dependances()
