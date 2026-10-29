"""Widgets partagés : cartes, panneau de métriques, visionneuse d'image, tableau des segments,
barre du dossier de sortie, boîtes de dialogue en français."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QFrame, QGridLayout, QHBoxLayout,
                               QHeaderView, QLabel, QMessageBox, QPlainTextEdit, QPushButton,
                               QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from core.environnement import ouvrir_dossier
from core.podcast import COULEURS_SEGMENTS, LIBELLES_SEGMENTS
from core.resultats import Metrique


# ----------------------------------------------------------------------
# Briques de mise en page
# ----------------------------------------------------------------------
class Carte(QFrame):
    """Carte blanche à coins arrondis avec un titre optionnel."""

    def __init__(self, titre: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("Carte")
        self._disposition = QVBoxLayout(self)
        self._disposition.setContentsMargins(20, 18, 20, 18)
        self._disposition.setSpacing(12)
        if titre:
            etiquette = QLabel(titre)
            etiquette.setObjectName("TitreCarte")
            self._disposition.addWidget(etiquette)

    @property
    def contenu(self) -> QVBoxLayout:
        return self._disposition


def etiquette_secondaire(texte: str) -> QLabel:
    lbl = QLabel(texte)
    lbl.setObjectName("TexteSecondaire")
    lbl.setWordWrap(True)
    return lbl


# ----------------------------------------------------------------------
# Boîtes de dialogue (boutons nommés en français : pas de boutons Qt standard en anglais)
# ----------------------------------------------------------------------
def informer(parent, titre: str, texte: str) -> None:
    boite = QMessageBox(parent)
    boite.setIcon(QMessageBox.Icon.Information)
    boite.setWindowTitle(titre)
    boite.setText(texte)
    boite.addButton("OK", QMessageBox.ButtonRole.AcceptRole)
    boite.exec()


def avertir(parent, titre: str, texte: str) -> None:
    boite = QMessageBox(parent)
    boite.setIcon(QMessageBox.Icon.Warning)
    boite.setWindowTitle(titre)
    boite.setText(texte)
    boite.addButton("OK", QMessageBox.ButtonRole.AcceptRole)
    boite.exec()


def demander_confirmation(parent, titre: str, texte: str, oui: str = "Oui", non: str = "Non") -> bool:
    boite = QMessageBox(parent)
    boite.setIcon(QMessageBox.Icon.Question)
    boite.setWindowTitle(titre)
    boite.setText(texte)
    bouton_oui = boite.addButton(oui, QMessageBox.ButtonRole.AcceptRole)
    boite.addButton(non, QMessageBox.ButtonRole.RejectRole)
    boite.exec()
    return boite.clickedButton() is bouton_oui


class DetailsTechniquesDialog(QDialog):
    """Fenêtre « Détails techniques » : trace complète de l'erreur."""

    def __init__(self, trace: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Détails techniques")
        self.resize(760, 460)
        disposition = QVBoxLayout(self)
        disposition.addWidget(etiquette_secondaire(
            "Copiez ce texte si vous devez signaler le problème."))
        zone = QPlainTextEdit(trace)
        zone.setReadOnly(True)
        disposition.addWidget(zone, 1)
        boutons = QHBoxLayout()
        boutons.addStretch(1)
        copier = QPushButton("Copier")
        copier.clicked.connect(lambda: zone.selectAll() or zone.copy())
        fermer = QPushButton("Fermer")
        fermer.clicked.connect(self.accept)
        boutons.addWidget(copier)
        boutons.addWidget(fermer)
        disposition.addLayout(boutons)


def afficher_erreur(parent, message: str, trace: str = "") -> None:
    """Message d'erreur lisible + bouton « Détails techniques » si une trace existe."""
    boite = QMessageBox(parent)
    boite.setIcon(QMessageBox.Icon.Critical)
    boite.setWindowTitle("Erreur")
    boite.setText(message)
    bouton_details = boite.addButton("Détails techniques", QMessageBox.ButtonRole.ActionRole) if trace else None
    boite.addButton("Fermer", QMessageBox.ButtonRole.RejectRole)
    boite.exec()
    if bouton_details is not None and boite.clickedButton() is bouton_details:
        DetailsTechniquesDialog(trace, parent).exec()


# ----------------------------------------------------------------------
# Métriques
# ----------------------------------------------------------------------
class PanneauMetriques(QFrame):
    """Grille de petites cartes ; chaque valeur est étiquetée « mesuré », « estimé » ou « info »
    pour ne jamais présenter une estimation comme une mesure exacte."""

    _TAGS = {"mesure": ("TagMesure", "mesuré"), "estime": ("TagEstime", "estimé"),
             "info": ("TagInfo", "info")}

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._grille = QGridLayout(self)
        self._grille.setContentsMargins(0, 0, 0, 0)
        self._grille.setSpacing(10)

    def definir(self, metriques: list[Metrique], colonnes: int = 3) -> None:
        while self._grille.count():
            item = self._grille.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for i, m in enumerate(metriques):
            carte = QFrame()
            carte.setObjectName("Metrique")
            v = QVBoxLayout(carte)
            v.setContentsMargins(12, 8, 12, 8)
            v.setSpacing(2)
            libelle = QLabel(m.libelle)
            libelle.setObjectName("LibelleMetrique")
            libelle.setWordWrap(True)
            valeur = QLabel(m.valeur)
            valeur.setObjectName("ValeurMetrique")
            valeur.setWordWrap(True)
            nom_style, texte_tag = self._TAGS.get(m.nature, self._TAGS["info"])
            tag = QLabel(texte_tag)
            tag.setObjectName(nom_style)
            tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ligne_tag = QHBoxLayout()
            ligne_tag.addWidget(tag)
            ligne_tag.addStretch(1)
            v.addWidget(libelle)
            v.addWidget(valeur)
            v.addLayout(ligne_tag)
            if m.aide:
                carte.setToolTip(m.aide)
            self._grille.addWidget(carte, i // colonnes, i % colonnes)


def widget_notes(notes: list[str]) -> Optional[QWidget]:
    """Remarques et limites du traitement, en texte secondaire."""
    if not notes:
        return None
    conteneur = QWidget()
    v = QVBoxLayout(conteneur)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(4)
    for note in notes:
        v.addWidget(etiquette_secondaire("• " + note))
    return conteneur


# ----------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------
class ImageDialog(QDialog):
    """Image en taille réelle dans une fenêtre redimensionnable et défilante."""

    def __init__(self, chemin: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(chemin.name)
        self.resize(1100, 700)
        disposition = QVBoxLayout(self)
        zone = QScrollArea()
        zone.setWidgetResizable(False)
        etiquette = QLabel()
        pixmap = QPixmap(str(chemin))
        if pixmap.isNull():
            etiquette.setText(f"Impossible d'afficher l'image : {chemin}")
        else:
            etiquette.setPixmap(pixmap)
        etiquette.adjustSize()
        zone.setWidget(etiquette)
        disposition.addWidget(zone, 1)
        boutons = QHBoxLayout()
        boutons.addStretch(1)
        systeme = QPushButton("Ouvrir avec le visualiseur du système")
        systeme.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(chemin))))
        fermer = QPushButton("Fermer")
        fermer.clicked.connect(self.accept)
        boutons.addWidget(systeme)
        boutons.addWidget(fermer)
        disposition.addLayout(boutons)


class ImageViewer(QFrame):
    """Image adaptée à la largeur disponible (redimensionnable), avec défilement si elle est
    haute, et un bouton pour l'ouvrir en taille réelle."""

    def __init__(self, titre: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._chemin: Optional[Path] = None
        self._pixmap = QPixmap()
        disposition = QVBoxLayout(self)
        disposition.setContentsMargins(0, 0, 0, 0)
        disposition.setSpacing(6)
        entete = QHBoxLayout()
        if titre:
            lbl = QLabel(titre)
            lbl.setObjectName("TitreCarte")
            entete.addWidget(lbl)
        entete.addStretch(1)
        self._bouton = QPushButton("Voir en taille réelle")
        self._bouton.clicked.connect(self._agrandir)
        entete.addWidget(self._bouton)
        disposition.addLayout(entete)
        self._zone = QScrollArea()
        self._zone.setWidgetResizable(True)
        self._zone.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._etiquette = QLabel()
        self._etiquette.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._zone.setWidget(self._etiquette)
        disposition.addWidget(self._zone)

    def charger(self, chemin) -> None:
        self._chemin = Path(chemin)
        self._pixmap = QPixmap(str(self._chemin))
        if self._pixmap.isNull():
            self._etiquette.setText(f"Impossible de charger l'image : {self._chemin}")
            self._bouton.setEnabled(False)
            self._zone.setMinimumHeight(60)
            return
        self._bouton.setEnabled(True)
        self._rafraichir()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rafraichir()

    def _rafraichir(self) -> None:
        if self._pixmap.isNull():
            return
        largeur = max(200, self._zone.viewport().width() - 6)
        adapte = self._pixmap.scaledToWidth(largeur, Qt.TransformationMode.SmoothTransformation)
        self._etiquette.setPixmap(adapte)
        self._zone.setMinimumHeight(min(adapte.height() + 6, 440))

    def _agrandir(self) -> None:
        if self._chemin is not None:
            ImageDialog(self._chemin, self).exec()


# ----------------------------------------------------------------------
# Segments (mode podcast)
# ----------------------------------------------------------------------
def _mm_ss(secondes: float) -> str:
    s = int(round(secondes))
    return f"{s // 60:02d}:{s % 60:02d}"


class TableauSegments(QTableWidget):
    """Tableau début / fin / type, avec les couleurs communes aux graphiques."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["Début", "Fin", "Type"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)

    def remplir(self, segments: list[tuple[float, float, str]]) -> None:
        self.setRowCount(len(segments))
        for ligne, (debut, fin, type_) in enumerate(segments):
            couleur = QColor(COULEURS_SEGMENTS.get(type_, "#cccccc"))
            couleur.setAlpha(110)
            for colonne, texte in enumerate((_mm_ss(debut), _mm_ss(fin), LIBELLES_SEGMENTS.get(type_, type_))):
                item = QTableWidgetItem(texte)
                item.setBackground(QBrush(couleur))
                self.setItem(ligne, colonne, item)
        self.setMinimumHeight(min(60 + 30 * len(segments), 320))


def legende_segments() -> QLabel:
    morceaux = [f'<span style="color:{COULEURS_SEGMENTS[t]}; font-size:16px;">■</span> '
                f'{LIBELLES_SEGMENTS[t]}' for t in ("parole", "musique", "mixte", "silence")]
    lbl = QLabel("&nbsp;&nbsp;&nbsp;".join(morceaux))
    lbl.setTextFormat(Qt.TextFormat.RichText)
    return lbl


# ----------------------------------------------------------------------
# Dossier de sortie
# ----------------------------------------------------------------------
class BarreDossierSortie(QFrame):
    """Affiche le dossier de résultats et propose de l'ouvrir."""

    def __init__(self, dossier: Path, fichiers: Optional[list[Path]] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._dossier = Path(dossier)
        disposition = QVBoxLayout(self)
        disposition.setContentsMargins(0, 0, 0, 0)
        disposition.setSpacing(6)
        if fichiers:
            liste = QLabel("Fichiers produits : " + ", ".join(f.name for f in fichiers))
            liste.setObjectName("TexteSecondaire")
            liste.setWordWrap(True)
            liste.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            disposition.addWidget(liste)
        ligne = QHBoxLayout()
        chemin = QLabel(f"Dossier : {self._dossier}")
        chemin.setWordWrap(True)
        chemin.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bouton = QPushButton("Ouvrir le dossier de sortie")
        bouton.clicked.connect(self._ouvrir)
        ligne.addWidget(chemin, 1)
        ligne.addWidget(bouton)
        disposition.addLayout(ligne)

    def _ouvrir(self) -> None:
        if not ouvrir_dossier(self._dossier):
            avertir(self, "Dossier introuvable",
                    f"Impossible d'ouvrir le dossier :\n{self._dossier}")
