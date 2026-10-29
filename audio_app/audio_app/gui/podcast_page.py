"""Page « Podcast / Radio » (mode B) : segmentation + isolation de la voix."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QComboBox, QLabel, QVBoxLayout

from core.demucs_separator import MODELE_PAR_DEFAUT
from core.environnement import demucs_disponible, message_installation_demucs
from core.pipelines import DUREE_MIN_SEPARATION_S, traiter_podcast
from core.resultats import ResultatTraitement
from gui.audio_player_widget import AudioPlayerWidget
from gui.page_base import PageTraitement, creer_combo_frequence
from gui.result_widgets import (ImageViewer, PanneauMetriques, TableauSegments,
                                etiquette_secondaire, legende_segments)


class PagePodcast(PageTraitement):
    TITRE = "Podcast / Radio"
    DESCRIPTION = ("Analysez un enregistrement, détectez les segments (parole, musique, mixte, silence) "
                   "et isolez la voix de la musique de fond.")
    TEXTE_LANCEMENT = "Analyser et isoler la voix"
    DUREE_MINIMALE_S = DUREE_MIN_SEPARATION_S
    MESSAGE_TROP_COURT = "Ce fichier est trop court pour analyser un podcast de façon fiable."

    def construire_parametres(self, disposition: QVBoxLayout) -> None:
        disposition.addWidget(QLabel("Moteur pour les passages mixtes (voix + musique)"))
        self._combo_moteur = QComboBox()
        self._combo_moteur.addItem("Demucs — Haute qualité (recommandé)", "demucs")
        self._combo_moteur.addItem("REPET-SIM — Rapide, qualité limitée", "repet")
        self._combo_moteur.currentIndexChanged.connect(self._maj_moteur)
        disposition.addWidget(self._combo_moteur)
        self._texte_moteur = etiquette_secondaire("")
        disposition.addWidget(self._texte_moteur)
        self._alerte_demucs = etiquette_secondaire("")
        disposition.addWidget(self._alerte_demucs)

        disposition.addWidget(QLabel("Fréquence de travail (REPET-SIM uniquement)"))
        self._combo_sr = creer_combo_frequence()
        disposition.addWidget(self._combo_sr)
        disposition.addWidget(etiquette_secondaire(
            "La segmentation utilise des seuils fixes (fenêtres d'une seconde). Seuls les passages « mixtes » "
            "sont séparés ; la parole seule et la musique seule gardent leur signal original."))
        self._maj_moteur()

    def _maj_moteur(self) -> None:
        demucs = self._combo_moteur.currentData() == "demucs"
        self._texte_moteur.setText(
            "Demucs (modèle htdemucs) traite uniquement les passages mixtes, avec fondus enchaînés aux "
            "frontières. Conçu pour la musique : la voix parlée peut être légèrement altérée. Sur processeur "
            "(CPU), le calcul peut durer plusieurs minutes ; premier lancement : téléchargement du modèle "
            "(Internet requis)." if demucs else
            "REPET-SIM sépare les passages mixtes en estimant la partie répétitive : rapide, mais la voix "
            "contient souvent encore de la musique.")
        self._combo_sr.setEnabled(not demucs)
        manquant = demucs and not demucs_disponible()
        self._alerte_demucs.setVisible(manquant)
        if manquant:
            self._alerte_demucs.setText("⚠ " + message_installation_demucs())

    def reinitialiser_parametres(self) -> None:
        self._combo_moteur.setCurrentIndex(0)
        self._combo_sr.setCurrentIndex(1)

    def valider_parametres(self) -> Optional[str]:
        if self._combo_moteur.currentData() == "demucs" and not demucs_disponible():
            return message_installation_demucs()
        return None

    def lancer_tache(self, dossier_sortie: Path) -> None:
        self._tache.lancer(traiter_podcast, self._fichier, self._combo_sr.currentData(), dossier_sortie,
                           moteur=self._combo_moteur.currentData(), modele=MODELE_PAR_DEFAUT)

    def construire_resultats(self, resultat: ResultatTraitement, disposition: QVBoxLayout) -> None:
        for cle, titre in (("voix", "Voix claire"), ("fond", "Musique de fond")):
            lecteur = AudioPlayerWidget(titre)
            lecteur.charger(resultat.fichiers[cle])
            disposition.addWidget(lecteur)
        panneau = PanneauMetriques()
        panneau.definir(resultat.metriques)
        disposition.addWidget(panneau)

        titre_segments = QLabel("Segments détectés")
        titre_segments.setObjectName("TitreCarte")
        disposition.addWidget(titre_segments)
        disposition.addWidget(legende_segments())
        tableau = TableauSegments()
        tableau.remplir(resultat.segments)
        disposition.addWidget(tableau)

        for cle, titre in (("segmentation", "Segmentation"), ("spectrogrammes", "Spectrogrammes")):
            vue = ImageViewer(titre)
            vue.charger(resultat.fichiers[cle])
            disposition.addWidget(vue)
