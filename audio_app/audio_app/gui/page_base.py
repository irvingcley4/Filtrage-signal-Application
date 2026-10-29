"""PageTraitement : socle commun des pages Musique, Restauration et Podcast.

Gère : choix du fichier, métadonnées, lecteur d'origine, dossier de sortie, validation,
lancement dans un worker, progression, annulation, erreurs, zone de résultats et états.
Les pages filles ne définissent que leurs paramètres, leur worker et leurs résultats.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QGridLayout, QHBoxLayout,
                               QLabel, QProgressBar, QPushButton, QScrollArea, QVBoxLayout, QWidget)

from core.audio_io import InfosAudio, lire_infos
from core.environnement import (creer_dossier_sortie, dossier_sortie_par_defaut,
                                ffmpeg_disponible)
from core.erreurs import AudioError
from core.resultats import ResultatTraitement
from gui.audio_player_widget import AudioPlayerWidget
from gui.result_widgets import (BarreDossierSortie, Carte, afficher_erreur, avertir,
                                etiquette_secondaire, widget_notes)
from gui.workers import ExecuteurTache

# États de la page
ETAT_AUCUN = "aucun"        # aucun fichier sélectionné
ETAT_CHARGE = "charge"      # fichier chargé, prêt
ETAT_EN_COURS = "en_cours"  # traitement en cours
ETAT_REUSSI = "reussi"      # résultat disponible
ETAT_ERREUR = "erreur"      # le dernier traitement a échoué

FILTRE_AUDIO = ("Fichiers audio (*.wav *.mp3 *.m4a *.flac *.ogg *.aac *.opus *.wma *.mpeg *.mpg);;"
                "Tous les fichiers (*)")


FREQUENCES_TRAVAIL = (16000, 22050, 32000, 44100)


def creer_combo_frequence() -> QComboBox:
    """Choix de la fréquence de travail (modes Musique et Podcast), 22050 Hz par défaut.
    Plus elle est haute, plus le calcul est long et gourmand en mémoire."""
    combo = QComboBox()
    for f in FREQUENCES_TRAVAIL:
        combo.addItem(f"{f} Hz" + (" (par défaut)" if f == 22050 else ""), f)
    combo.setCurrentIndex(FREQUENCES_TRAVAIL.index(22050))
    combo.setToolTip("Le fichier est rééchantillonné à cette fréquence avant traitement.")
    return combo


def _taille_lisible(octets: int) -> str:
    if octets >= 1_000_000:
        return f"{octets / 1_000_000:.1f} Mo"
    return f"{octets / 1000:.0f} Ko"


def _duree_lisible(secondes: float) -> str:
    s = int(round(secondes))
    return f"{s // 60} min {s % 60:02d} s ({secondes:.1f} s)"


class PageTraitement(QWidget):
    retour_accueil = Signal()

    TITRE = ""
    DESCRIPTION = ""
    TEXTE_LANCEMENT = "Lancer le traitement"
    DUREE_MINIMALE_S = 5.0
    MESSAGE_TROP_COURT = "Ce fichier est trop court pour réaliser un traitement fiable."

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("Page")
        self._fichier: Optional[Path] = None
        self._infos: Optional[InfosAudio] = None
        self._resultat: Optional[ResultatTraitement] = None
        self._dossier_sortie: Path = dossier_sortie_par_defaut()
        self._etat = ETAT_AUCUN
        self._etat_fin = ETAT_CHARGE

        self._tache = ExecuteurTache(self)
        self._tache.statut.connect(self._sur_statut)
        self._tache.progression.connect(self._sur_progression)
        self._tache.termine.connect(self._sur_termine)
        self._tache.erreur.connect(self._sur_erreur)
        self._tache.annule.connect(self._sur_annulation)
        self._tache.fini.connect(self._sur_fini)

        self._construire_interface()
        self._definir_etat(ETAT_AUCUN)
        self._poser_statut("Aucun fichier sélectionné.", "info")

    # ==================================================================
    # Points d'extension pour les pages filles
    # ==================================================================
    def construire_parametres(self, disposition: QVBoxLayout) -> None:
        """Ajoute les widgets de paramètres dans la carte « Paramètres »."""

    def reinitialiser_parametres(self) -> None:
        """Remet les paramètres à leurs valeurs par défaut."""

    def valider_parametres(self) -> Optional[str]:
        """Retourne un message d'erreur en français, ou None si les paramètres sont valides."""
        return None

    def lancer_tache(self, dossier_sortie: Path) -> None:
        """Appelle self._tache.lancer(fonction_de_core, ...)."""
        raise NotImplementedError

    def construire_resultats(self, resultat: ResultatTraitement, disposition: QVBoxLayout) -> None:
        """Ajoute les widgets de résultats (lecteurs, métriques, images…)."""
        raise NotImplementedError

    # ==================================================================
    # Construction de l'interface
    # ==================================================================
    def _construire_interface(self) -> None:
        racine = QVBoxLayout(self)
        racine.setContentsMargins(0, 0, 0, 0)
        defilement = QScrollArea()
        defilement.setWidgetResizable(True)
        contenu = QWidget()
        contenu.setObjectName("Contenu")
        colonne = QVBoxLayout(contenu)
        colonne.setContentsMargins(28, 24, 28, 28)
        colonne.setSpacing(16)
        defilement.setWidget(contenu)
        racine.addWidget(defilement)

        # --- en-tête ---
        entete = QHBoxLayout()
        retour = QPushButton("← Retour")
        retour.setToolTip("Revenir à l'accueil (vos fichiers et résultats sont conservés)")
        retour.clicked.connect(self.retour_accueil.emit)
        entete.addWidget(retour)
        entete.addStretch(1)
        colonne.addLayout(entete)
        titre = QLabel(self.TITRE)
        titre.setObjectName("TitrePage")
        colonne.addWidget(titre)
        colonne.addWidget(etiquette_secondaire(self.DESCRIPTION))

        # --- 1. fichier ---
        self._carte_fichier = Carte("1. Fichier audio")
        ligne = QHBoxLayout()
        self._btn_choisir = QPushButton("Choisir un fichier audio")
        self._btn_choisir.setProperty("primaire", True)
        self._btn_choisir.setToolTip("Formats : WAV, MP3, M4A, FLAC, OGG, AAC… (FFmpeg requis hors WAV)")
        self._btn_choisir.clicked.connect(self.choisir_fichier)
        ligne.addWidget(self._btn_choisir)
        ligne.addStretch(1)
        self._carte_fichier.contenu.addLayout(ligne)
        grille = QGridLayout()
        grille.setHorizontalSpacing(14)
        self._champs: dict[str, QLabel] = {}
        for i, nom in enumerate(("Nom", "Chemin", "Durée", "Fréquence d'échantillonnage", "Taille")):
            etiquette = QLabel(nom + " :")
            etiquette.setObjectName("TexteSecondaire")
            valeur = QLabel("—")
            valeur.setWordWrap(True)
            valeur.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            grille.addWidget(etiquette, i, 0, Qt.AlignmentFlag.AlignTop)
            grille.addWidget(valeur, i, 1)
            self._champs[nom] = valeur
        grille.setColumnStretch(1, 1)
        self._carte_fichier.contenu.addLayout(grille)
        self._lecteur_original = AudioPlayerWidget("Écoute du fichier original")
        self._carte_fichier.contenu.addWidget(self._lecteur_original)
        colonne.addWidget(self._carte_fichier)

        # --- 2. paramètres ---
        self._carte_params = Carte("2. Paramètres")
        self.construire_parametres(self._carte_params.contenu)
        colonne.addWidget(self._carte_params)

        # --- 3. dossier de sortie ---
        self._carte_sortie = Carte("3. Dossier de sortie")
        ligne_sortie = QHBoxLayout()
        self._label_dossier = QLabel()
        self._label_dossier.setWordWrap(True)
        self._btn_dossier = QPushButton("Changer…")
        self._btn_dossier.clicked.connect(self.choisir_dossier_sortie)
        ligne_sortie.addWidget(self._label_dossier, 1)
        ligne_sortie.addWidget(self._btn_dossier)
        self._carte_sortie.contenu.addLayout(ligne_sortie)
        self._carte_sortie.contenu.addWidget(etiquette_secondaire(
            "Chaque traitement crée un sous-dossier horodaté (resultats_AAAA-MM-JJ_HHMM) : "
            "aucun fichier existant n'est écrasé."))
        self._maj_label_dossier()
        colonne.addWidget(self._carte_sortie)

        # --- 4. traitement ---
        self._carte_action = Carte("4. Traitement")
        ligne_action = QHBoxLayout()
        self._btn_lancer = QPushButton(self.TEXTE_LANCEMENT)
        self._btn_lancer.setProperty("primaire", True)
        self._btn_lancer.clicked.connect(self.lancer)
        self._btn_annuler = QPushButton("Annuler")
        self._btn_annuler.setToolTip("Interrompt le traitement à la prochaine étape")
        self._btn_annuler.clicked.connect(self._demander_annulation)
        ligne_action.addWidget(self._btn_lancer)
        ligne_action.addWidget(self._btn_annuler)
        ligne_action.addStretch(1)
        self._carte_action.contenu.addLayout(ligne_action)
        self._barre = QProgressBar()
        self._barre.setTextVisible(False)
        self._carte_action.contenu.addWidget(self._barre)
        self._label_statut = QLabel()
        self._label_statut.setObjectName("Statut")
        self._label_statut.setWordWrap(True)
        self._carte_action.contenu.addWidget(self._label_statut)
        colonne.addWidget(self._carte_action)

        # --- 5. résultats ---
        self._zone_resultats = Carte("Résultats")
        self._zone_resultats.hide()
        colonne.addWidget(self._zone_resultats)

        # --- pied de page ---
        pied = QHBoxLayout()
        self._btn_recommencer = QPushButton("Recommencer")
        self._btn_recommencer.setToolTip("Efface le fichier et les résultats de cette page")
        self._btn_recommencer.clicked.connect(self.reinitialiser)
        accueil = QPushButton("Retour à l'accueil")
        accueil.clicked.connect(self.retour_accueil.emit)
        pied.addWidget(self._btn_recommencer)
        pied.addStretch(1)
        pied.addWidget(accueil)
        colonne.addLayout(pied)
        colonne.addStretch(1)

    # ==================================================================
    # Fichier et dossier
    # ==================================================================
    def choisir_fichier(self) -> None:
        depart = str(self._fichier.parent) if self._fichier else str(Path.home())
        chemin, _ = QFileDialog.getOpenFileName(self, "Choisir un fichier audio", depart, FILTRE_AUDIO)
        if chemin:
            self.charger_fichier(Path(chemin))

    def charger_fichier(self, chemin: Path) -> None:
        """Lit les métadonnées (en-tête WAV ou ffprobe : rapide) et valide le fichier."""
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        message = None
        infos = None
        try:
            infos = lire_infos(chemin)
        except AudioError as exc:
            message = str(exc)
        finally:
            QApplication.restoreOverrideCursor()
        if infos is None:
            avertir(self, "Fichier audio invalide", message or "Fichier illisible.")
            return
        self._fichier, self._infos = chemin, infos
        self._champs["Nom"].setText(infos.nom)
        self._champs["Chemin"].setText(str(infos.chemin))
        self._champs["Durée"].setText(_duree_lisible(infos.duree_s))
        self._champs["Fréquence d'échantillonnage"].setText(f"{infos.frequence} Hz")
        self._champs["Taille"].setText(_taille_lisible(infos.taille_octets))
        self._lecteur_original.charger(chemin)
        self._effacer_resultats()
        self._definir_etat(ETAT_CHARGE)
        self._poser_statut("Fichier chargé. Prêt à lancer le traitement.", "info")

    def choisir_dossier_sortie(self) -> None:
        dossier = QFileDialog.getExistingDirectory(self, "Choisir le dossier de sortie",
                                                   str(self._dossier_sortie))
        if dossier:
            self._dossier_sortie = Path(dossier)
            self._maj_label_dossier()

    def _maj_label_dossier(self) -> None:
        self._label_dossier.setText(str(self._dossier_sortie))

    # ==================================================================
    # Lancement et suivi
    # ==================================================================
    def _valider(self) -> Optional[str]:
        if self._fichier is None or self._infos is None:
            return "Veuillez sélectionner un fichier audio avant de lancer le traitement."
        if not self._fichier.is_file():
            return f"Le fichier n'existe plus : {self._fichier}"
        if self._infos.duree_s < self.DUREE_MINIMALE_S:
            return self.MESSAGE_TROP_COURT
        if self._fichier.suffix.lower() != ".wav" and not ffmpeg_disponible():
            return ("FFmpeg est requis pour lire ce format audio. Installez-le (voir le README) "
                    "puis relancez l'application, ou convertissez le fichier en WAV.")
        return self.valider_parametres()

    def lancer(self) -> None:
        if self._tache.en_cours():
            return
        probleme = self._valider()
        if probleme:
            avertir(self, "Vérification avant traitement", probleme)
            return
        try:
            dossier = creer_dossier_sortie(self._dossier_sortie)
        except OSError as exc:
            avertir(self, "Dossier de sortie",
                    f"Impossible de créer le dossier de résultats dans :\n{self._dossier_sortie}\n\n{exc}")
            return
        self._effacer_resultats()
        self._etat_fin = ETAT_CHARGE
        self._definir_etat(ETAT_EN_COURS)
        self._poser_statut("Démarrage du traitement…", "info")
        self._barre.setRange(0, 0)           # indéterminée tant qu'aucune étape n'est signalée
        self.lancer_tache(dossier)

    def _demander_annulation(self) -> None:
        self._btn_annuler.setEnabled(False)
        self._poser_statut("Annulation en cours (à la prochaine étape)…", "info")
        self._tache.demander_annulation()

    def _sur_statut(self, texte: str) -> None:
        self._poser_statut(texte, "info")

    def _sur_progression(self, valeur: float) -> None:
        if valeur < 0:                       # avancement inconnu : barre indéterminée
            self._barre.setRange(0, 0)
        else:
            self._barre.setRange(0, 1000)
            self._barre.setValue(int(max(0.0, min(1.0, valeur)) * 1000))

    def _sur_termine(self, resultat: ResultatTraitement) -> None:
        self._resultat = resultat
        self._barre.setRange(0, 1000)
        self._barre.setValue(1000)
        try:
            self._afficher_resultat(resultat)
        except Exception:                     # erreur d'affichage : ne jamais laisser l'UI bloquée
            import traceback
            self._etat_fin = ETAT_ERREUR
            self._poser_statut("Le traitement est terminé, mais l'affichage des résultats a échoué.", "erreur")
            afficher_erreur(self, "L'affichage des résultats a échoué.", traceback.format_exc())
            return
        self._etat_fin = ETAT_REUSSI
        self._poser_statut(f"Le traitement est terminé avec succès "
                           f"({resultat.duree_calcul_s:.1f} s de calcul).", "succes")

    def _sur_erreur(self, message: str, trace: str) -> None:
        self._etat_fin = ETAT_ERREUR
        self._poser_statut(message, "erreur")
        afficher_erreur(self, message, trace)

    def _sur_annulation(self) -> None:
        self._etat_fin = ETAT_CHARGE
        self._poser_statut("Traitement annulé.", "info")

    def _sur_fini(self) -> None:
        """Le thread est réellement terminé : on peut rétablir les boutons."""
        self._definir_etat(self._etat_fin)

    # ==================================================================
    # Résultats
    # ==================================================================
    def _afficher_resultat(self, resultat: ResultatTraitement) -> None:
        self._effacer_resultats()          # remet aussi self._resultat à None...
        self._resultat = resultat          # ...donc on le réaffecte ensuite (utilisé par « Exporter »)
        self.construire_resultats(resultat, self._zone_resultats.contenu)
        notes = widget_notes(resultat.notes)
        if notes is not None:
            self._zone_resultats.contenu.addWidget(notes)
        fichiers = [p for cle, p in resultat.fichiers.items() if cle != "original"]
        self._zone_resultats.contenu.addWidget(BarreDossierSortie(resultat.dossier, fichiers))
        self._zone_resultats.show()

    def _effacer_resultats(self) -> None:
        """Retire les widgets de résultats (en libérant leurs lecteurs audio)."""
        self._resultat = None
        disposition = self._zone_resultats.contenu
        while disposition.count() > 1:        # l'élément 0 est le titre « Résultats »
            item = disposition.takeAt(1)
            widget = item.widget()
            if widget is not None:
                for lecteur in widget.findChildren(AudioPlayerWidget):
                    lecteur.liberer()
                if isinstance(widget, AudioPlayerWidget):
                    widget.liberer()
                widget.setParent(None)
                widget.deleteLater()
            elif item.layout() is not None:
                item.layout().deleteLater()
        self._zone_resultats.hide()

    # ==================================================================
    # États
    # ==================================================================
    def _poser_statut(self, texte: str, type_: str) -> None:
        self._label_statut.setText(texte)
        self._label_statut.setProperty("etat", type_)
        style = self._label_statut.style()
        style.unpolish(self._label_statut)    # force Qt à ré-appliquer la feuille de style
        style.polish(self._label_statut)

    def _definir_etat(self, etat: str) -> None:
        self._etat = etat
        en_cours = etat == ETAT_EN_COURS
        a_fichier = self._fichier is not None
        self._btn_choisir.setEnabled(not en_cours)
        self._carte_params.setEnabled(not en_cours)
        self._btn_dossier.setEnabled(not en_cours)
        self._btn_lancer.setEnabled(a_fichier and not en_cours)
        self._btn_annuler.setVisible(en_cours)
        self._btn_annuler.setEnabled(en_cours)
        self._btn_recommencer.setEnabled(not en_cours)
        self._barre.setVisible(en_cours or etat == ETAT_REUSSI)

    def reinitialiser(self) -> None:
        """« Recommencer » : efface fichier, résultats et paramètres de cette page."""
        if self._tache.en_cours():
            return
        self._lecteur_original.charger(None)
        self._effacer_resultats()
        self._fichier, self._infos = None, None
        for champ in self._champs.values():
            champ.setText("—")
        self.reinitialiser_parametres()
        self._definir_etat(ETAT_AUCUN)
        self._poser_statut("Aucun fichier sélectionné.", "info")

    # ==================================================================
    # Cycle de vie (appelé par la fenêtre principale)
    # ==================================================================
    def tache_en_cours(self) -> bool:
        return self._tache.en_cours()

    def demander_annulation(self) -> None:
        self._tache.demander_annulation()

    def attendre_fin(self, delai_ms: int) -> bool:
        return self._tache.attendre_fin(delai_ms)

    def liberer_ressources(self) -> None:
        self._lecteur_original.liberer()
        for lecteur in self._zone_resultats.findChildren(AudioPlayerWidget):
            lecteur.liberer()
