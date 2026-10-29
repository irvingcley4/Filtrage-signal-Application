"""Page « Séparer une musique » (mode A) : REPET-SIM ou NMF."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QComboBox, QLabel, QVBoxLayout

from core.demucs_separator import MODELES
from core.environnement import demucs_disponible, dependance_disponible, message_installation_demucs
from core.pipelines import DUREE_MIN_SEPARATION_S, traiter_musique
from core.resultats import ResultatTraitement
from gui.audio_player_widget import AudioPlayerWidget
from gui.page_base import PageTraitement, creer_combo_frequence
from gui.result_widgets import ImageViewer, PanneauMetriques, etiquette_secondaire

_TEXTES_METHODE = {
    "demucs": "Recommandée. Réseau de neurones (Demucs) entraîné sur de la musique : sépare voix, batterie, "
              "basse et autres instruments, puis reconstruit l'instrumental. Sur processeur (CPU), le "
              "traitement peut prendre plusieurs minutes selon la durée du morceau ; un GPU NVIDIA est "
              "nettement plus rapide. Premier lancement : téléchargement du modèle (connexion Internet requise).",
    "repet": "Rapide, qualité limitée. Estime la partie qui se RÉPÈTE (batterie, basse, boucles) : "
             "ce qui reste est la voix. Fonctionne bien quand l'instrumental boucle.",
    "nmf": "Expérimental, qualité limitée. Factorise le spectrogramme en « briques » puis les classe voix / "
           "instrumental avec une règle simple : moins fiable, mais montre le principe. "
           "Nécessite scikit-learn ; l'avancement ne peut pas être mesuré.",
}


class PageMusique(PageTraitement):
    TITRE = "Séparer une musique"
    DESCRIPTION = "Isolez la voix et l'instrumental d'un morceau de musique."
    TEXTE_LANCEMENT = "Lancer la séparation"
    DUREE_MINIMALE_S = DUREE_MIN_SEPARATION_S
    MESSAGE_TROP_COURT = "Ce fichier est trop court pour réaliser une séparation fiable."

    def construire_parametres(self, disposition: QVBoxLayout) -> None:
        disposition.addWidget(QLabel("Méthode de séparation"))
        self._combo_methode = QComboBox()
        self._combo_methode.addItem("Demucs — Haute qualité (recommandé)", "demucs")
        self._combo_methode.addItem("REPET-SIM — Rapide, qualité limitée", "repet")
        self._combo_methode.addItem("NMF — Expérimental, qualité limitée", "nmf")
        self._combo_methode.currentIndexChanged.connect(self._maj_methode)
        disposition.addWidget(self._combo_methode)
        self._texte_methode = etiquette_secondaire("")
        disposition.addWidget(self._texte_methode)
        self._alerte_demucs = etiquette_secondaire("")
        disposition.addWidget(self._alerte_demucs)

        disposition.addWidget(QLabel("Modèle Demucs"))
        self._combo_modele = QComboBox()
        for nom, description in MODELES.items():
            self._combo_modele.addItem(description, nom)
        self._combo_modele.setToolTip("htdemucs_ft peut être plus lent et n'est pas forcément meilleur sur tous les morceaux.")
        disposition.addWidget(self._combo_modele)

        disposition.addWidget(QLabel("Fréquence de travail (REPET-SIM et NMF uniquement)"))
        self._combo_sr = creer_combo_frequence()
        disposition.addWidget(self._combo_sr)
        self._texte_sr = etiquette_secondaire(
            "22050 Hz suffit pour la voix et accélère le calcul ; une fréquence plus haute "
            "conserve plus d'aigus mais demande plus de temps et de mémoire. Demucs travaille toujours "
            "à 44100 Hz (stéréo).")
        disposition.addWidget(self._texte_sr)
        self._maj_methode()

    def _maj_methode(self) -> None:
        methode = self._combo_methode.currentData()
        self._texte_methode.setText(_TEXTES_METHODE[methode])
        self._combo_modele.setEnabled(methode == "demucs")      # modèle : Demucs seulement
        self._combo_sr.setEnabled(methode != "demucs")          # fréquence : REPET-SIM / NMF seulement
        manquant = methode == "demucs" and not demucs_disponible()
        self._alerte_demucs.setVisible(manquant)
        if manquant:
            self._alerte_demucs.setText("⚠ " + message_installation_demucs())

    def reinitialiser_parametres(self) -> None:
        self._combo_methode.setCurrentIndex(0)
        self._combo_modele.setCurrentIndex(0)
        self._combo_sr.setCurrentIndex(1)

    def valider_parametres(self) -> Optional[str]:
        methode = self._combo_methode.currentData()
        if methode == "demucs" and not demucs_disponible():
            return message_installation_demucs()
        if methode == "nmf" and not dependance_disponible("sklearn"):
            return "La dépendance scikit-learn est nécessaire pour utiliser NMF."
        return None

    def lancer_tache(self, dossier_sortie: Path) -> None:
        self._tache.lancer(traiter_musique, self._fichier, self._combo_methode.currentData(),
                           self._combo_sr.currentData(), dossier_sortie,
                           modele=self._combo_modele.currentData())

    def construire_resultats(self, resultat: ResultatTraitement, disposition: QVBoxLayout) -> None:
        for cle, titre in (("voix", "Voix isolée"), ("instrumental", "Instrumental")):
            lecteur = AudioPlayerWidget(titre)
            lecteur.charger(resultat.fichiers[cle])
            disposition.addWidget(lecteur)
        panneau = PanneauMetriques()
        panneau.definir(resultat.metriques)
        disposition.addWidget(panneau)
        vue = ImageViewer("Spectrogrammes")
        vue.charger(resultat.fichiers["spectrogrammes"])
        disposition.addWidget(vue)
