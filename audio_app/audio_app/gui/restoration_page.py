"""Page « Nettoyer une voix » (mode C) : passe-haut dynamique + soustraction spectrale (alpha)."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QPushButton,
                               QSlider, QVBoxLayout)

from core.pipelines import DUREE_MIN_RESTAURATION_S, traiter_restauration
from core.resultats import ResultatTraitement
from gui.audio_player_widget import AudioPlayerWidget
from gui.page_base import PageTraitement
from gui.result_widgets import (ImageViewer, PanneauMetriques, avertir, etiquette_secondaire,
                                informer)

ALPHA_DEFAUT = 2.0


class PageRestauration(PageTraitement):
    TITRE = "Nettoyer une voix"
    DESCRIPTION = ("Réduisez le bruit, le souffle et les basses fréquences indésirables (vent, "
                   "grondements) d'un enregistrement vocal.")
    TEXTE_LANCEMENT = "Nettoyer la voix"
    DUREE_MINIMALE_S = DUREE_MIN_RESTAURATION_S
    MESSAGE_TROP_COURT = "Ce fichier est trop court pour nettoyer la voix de façon fiable."

    def construire_parametres(self, disposition: QVBoxLayout) -> None:
        disposition.addWidget(QLabel("Force de la réduction de bruit (alpha)"))
        ligne = QHBoxLayout()
        self._curseur = QSlider(Qt.Orientation.Horizontal)
        self._curseur.setRange(10, 30)                 # 1,0 à 3,0 par pas de 0,1
        self._curseur.setValue(int(ALPHA_DEFAUT * 10))
        self._champ = QDoubleSpinBox()
        self._champ.setRange(1.0, 3.0)
        self._champ.setSingleStep(0.1)
        self._champ.setDecimals(1)
        self._champ.setValue(ALPHA_DEFAUT)
        self._curseur.valueChanged.connect(self._depuis_curseur)
        self._champ.valueChanged.connect(self._depuis_champ)
        ligne.addWidget(self._curseur, 1)
        ligne.addWidget(self._champ)
        disposition.addLayout(ligne)
        self._explication = etiquette_secondaire("")
        disposition.addWidget(self._explication)
        disposition.addWidget(etiquette_secondaire(
            "1,0 : réduction douce du bruit  ·  2,0 : compromis entre réduction du bruit et qualité  ·  "
            "3,0 : réduction plus forte, avec un risque accru d'artefacts (sons « métalliques »)."))
        disposition.addWidget(etiquette_secondaire(
            "La fréquence de coupure du filtre passe-haut (100 à 120 Hz) est choisie automatiquement "
            "d'après le fichier ; sa valeur effective sera affichée dans les résultats."))
        self._maj_explication(ALPHA_DEFAUT)

    def _depuis_curseur(self, valeur: int) -> None:
        self._champ.blockSignals(True)
        self._champ.setValue(valeur / 10.0)
        self._champ.blockSignals(False)
        self._maj_explication(valeur / 10.0)

    def _depuis_champ(self, valeur: float) -> None:
        self._curseur.blockSignals(True)
        self._curseur.setValue(int(round(valeur * 10)))
        self._curseur.blockSignals(False)
        self._maj_explication(valeur)

    def _maj_explication(self, alpha: float) -> None:
        if alpha < 1.5:
            texte = "Réglage actuel : réduction douce."
        elif alpha <= 2.5:
            texte = "Réglage actuel : compromis entre bruit et qualité."
        else:
            texte = "Réglage actuel : réduction forte, artefacts possibles."
        self._explication.setText(texte)

    def alpha(self) -> float:
        return round(self._champ.value(), 1)

    def reinitialiser_parametres(self) -> None:
        self._champ.setValue(ALPHA_DEFAUT)

    def valider_parametres(self) -> Optional[str]:
        if not 1.0 <= self.alpha() <= 3.0:
            return "Le paramètre alpha doit être compris entre 1,0 et 3,0."
        return None

    def lancer_tache(self, dossier_sortie: Path) -> None:
        self._tache.lancer(traiter_restauration, self._fichier, self.alpha(), dossier_sortie)

    def construire_resultats(self, resultat: ResultatTraitement, disposition: QVBoxLayout) -> None:
        avant = AudioPlayerWidget("Avant traitement")
        avant.charger(resultat.fichiers["original"])
        apres = AudioPlayerWidget("Après traitement")
        apres.charger(resultat.fichiers["restaure"])
        disposition.addWidget(avant)
        disposition.addWidget(apres)
        panneau = PanneauMetriques()
        panneau.definir(resultat.metriques)
        disposition.addWidget(panneau)
        vue = ImageViewer("Spectrogrammes avant / après")
        vue.charger(resultat.fichiers["spectrogrammes"])
        disposition.addWidget(vue)
        exporter = QPushButton("Exporter la voix restaurée…")
        exporter.clicked.connect(self._exporter)
        ligne = QHBoxLayout()
        ligne.addWidget(exporter)
        ligne.addStretch(1)
        disposition.addLayout(ligne)

    def _exporter(self) -> None:
        if self._resultat is None:
            return
        source = self._resultat.fichiers["restaure"]
        chemin, _ = QFileDialog.getSaveFileName(
            self, "Exporter la voix restaurée", str(source.parent / "voix_restauree.wav"),
            "Fichier WAV (*.wav)")
        if not chemin:
            return
        cible = Path(chemin)
        if cible.suffix.lower() != ".wav":
            cible = cible.with_suffix(".wav")
        try:
            shutil.copy2(source, cible)
        except OSError as exc:
            avertir(self, "Export impossible", f"Le fichier n'a pas pu être exporté :\n{exc}")
        else:
            informer(self, "Export terminé", f"Fichier exporté :\n{cible}")
