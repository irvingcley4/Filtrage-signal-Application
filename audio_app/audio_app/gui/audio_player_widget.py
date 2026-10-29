"""AudioPlayerWidget : lecteur audio réutilisable (QMediaPlayer + QAudioOutput).

Un seul lecteur joue à la fois : lancer la lecture d'un lecteur met les autres en pause.
Si le système ne sait pas lire le format (codec manquant), le fichier est converti en WAV
temporaire par FFmpeg (le fichier original n'est jamais modifié).
"""
from __future__ import annotations

import weakref
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QTemporaryDir, Qt, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton, QSlider,
                               QStyle, QVBoxLayout)

from core.audio_io import convertir_pour_lecture
from core.environnement import ffmpeg_disponible
from core.erreurs import AudioError


def _formater_temps(ms: int) -> str:
    secondes = max(0, int(ms // 1000))
    return f"{secondes // 60:02d}:{secondes % 60:02d}"


class AudioPlayerWidget(QFrame):
    _instances: "weakref.WeakSet[AudioPlayerWidget]" = weakref.WeakSet()

    def __init__(self, titre: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("LecteurAudio")
        self._source: Optional[Path] = None
        self._conversion_tentee = False
        self._dossier_temp: Optional[QTemporaryDir] = None

        self._sortie = QAudioOutput(self)
        self._sortie.setVolume(0.8)
        self._lecteur = QMediaPlayer(self)
        self._lecteur.setAudioOutput(self._sortie)

        # --- interface ---
        racine = QVBoxLayout(self)
        racine.setContentsMargins(14, 10, 14, 10)
        racine.setSpacing(6)

        self._titre = QLabel(titre)
        self._titre.setObjectName("TitreCarte")
        self._titre.setVisible(bool(titre))
        racine.addWidget(self._titre)

        ligne = QHBoxLayout()
        ligne.setSpacing(8)
        style = self.style()
        self._btn_lecture = QPushButton("Lecture")
        self._btn_lecture.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self._btn_lecture.setToolTip("Lecture / Pause")
        self._btn_lecture.clicked.connect(self._lecture_pause)
        self._btn_stop = QPushButton("Arrêt")
        self._btn_stop.setIcon(style.standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self._btn_stop.setToolTip("Arrêter et revenir au début")
        self._btn_stop.clicked.connect(self.arreter)

        self._curseur = QSlider(Qt.Orientation.Horizontal)
        self._curseur.setRange(0, 0)
        self._curseur.setToolTip("Position de lecture")
        self._curseur.sliderMoved.connect(self._lecteur.setPosition)
        self._curseur.sliderReleased.connect(lambda: self._lecteur.setPosition(self._curseur.value()))

        self._temps = QLabel("00:00 / 00:00")
        self._temps.setObjectName("TexteSecondaire")

        self._volume = QSlider(Qt.Orientation.Horizontal)
        self._volume.setRange(0, 100)
        self._volume.setValue(80)
        self._volume.setFixedWidth(90)
        self._volume.setToolTip("Volume")
        self._volume.valueChanged.connect(lambda v: self._sortie.setVolume(v / 100.0))

        ligne.addWidget(self._btn_lecture)
        ligne.addWidget(self._btn_stop)
        ligne.addWidget(self._curseur, 1)
        ligne.addWidget(self._temps)
        ligne.addWidget(QLabel("Vol."))
        ligne.addWidget(self._volume)
        racine.addLayout(ligne)

        self._erreur = QLabel("")
        self._erreur.setObjectName("ErreurLecteur")
        self._erreur.setWordWrap(True)
        self._erreur.hide()
        racine.addWidget(self._erreur)

        # --- signaux du lecteur ---
        self._lecteur.positionChanged.connect(self._sur_position)
        self._lecteur.durationChanged.connect(self._sur_duree)
        self._lecteur.playbackStateChanged.connect(self._sur_etat)
        self._lecteur.errorOccurred.connect(self._sur_erreur)

        self._activer(False)
        AudioPlayerWidget._instances.add(self)

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------
    def charger(self, chemin) -> None:
        """Charge un fichier (None = vider le lecteur)."""
        self._lecteur.stop()
        self._erreur.hide()
        self._conversion_tentee = False
        self._curseur.setValue(0)
        if chemin is None:
            self._source = None
            self._lecteur.setSource(QUrl())
            self._temps.setText("00:00 / 00:00")
            self._activer(False)
            return
        self._source = Path(chemin)
        if not self._source.is_file():
            self._afficher_erreur(f"Fichier introuvable : {self._source}")
            self._activer(False)
            return
        self._activer(True)
        self._lecteur.setSource(QUrl.fromLocalFile(str(self._source)))

    def arreter(self) -> None:
        self._lecteur.stop()

    def liberer(self) -> None:
        """Arrête la lecture et libère le fichier et les ressources temporaires."""
        try:
            self._lecteur.stop()
            self._lecteur.setSource(QUrl())
        except RuntimeError:        # objet Qt déjà détruit
            return
        self._dossier_temp = None   # QTemporaryDir supprime son dossier à la destruction

    # ------------------------------------------------------------------
    # Interne
    # ------------------------------------------------------------------
    def _activer(self, actif: bool) -> None:
        for w in (self._btn_lecture, self._btn_stop, self._curseur, self._volume):
            w.setEnabled(actif)

    def _afficher_erreur(self, texte: str) -> None:
        self._erreur.setText(texte)
        self._erreur.show()

    def _mettre_en_pause(self) -> None:
        try:
            if self._lecteur.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self._lecteur.pause()
        except RuntimeError:
            pass

    def _lecture_pause(self) -> None:
        if self._lecteur.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._lecteur.pause()
            return
        for autre in list(AudioPlayerWidget._instances):    # un seul lecteur actif à la fois
            if autre is not self:
                autre._mettre_en_pause()
        self._lecteur.play()

    def _sur_position(self, position: int) -> None:
        if not self._curseur.isSliderDown():
            self._curseur.setValue(position)
        self._temps.setText(f"{_formater_temps(position)} / {_formater_temps(self._lecteur.duration())}")

    def _sur_duree(self, duree: int) -> None:
        self._curseur.setRange(0, max(0, duree))
        self._temps.setText(f"{_formater_temps(self._lecteur.position())} / {_formater_temps(duree)}")

    def _sur_etat(self, etat) -> None:
        lecture = etat == QMediaPlayer.PlaybackState.PlayingState
        icone = QStyle.StandardPixmap.SP_MediaPause if lecture else QStyle.StandardPixmap.SP_MediaPlay
        self._btn_lecture.setIcon(self.style().standardIcon(icone))
        self._btn_lecture.setText("Pause" if lecture else "Lecture")

    def _sur_erreur(self, erreur, texte: str) -> None:
        """Format non lisible par le système : on tente une fois une conversion WAV par FFmpeg."""
        if not self._conversion_tentee and self._source is not None and ffmpeg_disponible():
            self._conversion_tentee = True
            try:
                if self._dossier_temp is None:
                    self._dossier_temp = QTemporaryDir()
                wav = convertir_pour_lecture(self._source, Path(self._dossier_temp.path()))
            except AudioError as exc:
                self._afficher_erreur(str(exc))
                return
            self._erreur.hide()
            self._lecteur.setSource(QUrl.fromLocalFile(str(wav)))
            return
        self._afficher_erreur("Lecture impossible : " + (texte or "format non pris en charge par ce système")
                              + ("" if ffmpeg_disponible() else " (installer FFmpeg peut résoudre le problème)"))
