"""Page « Démonstrations » : deux démos sur signaux synthétiques, sans fichier audio.
Elles exécutent les VRAIS algorithmes ; les SNR sont réellement mesurés (la vérité est connue)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel, QProgressBar,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)

from core.demos import executer_demo_musique, executer_demo_restauration
from core.environnement import (creer_dossier_sortie, dependance_disponible,
                                dossier_sortie_par_defaut)
from core.resultats import ResultatTraitement
from gui.audio_player_widget import AudioPlayerWidget
from gui.result_widgets import (BarreDossierSortie, Carte, ImageViewer, PanneauMetriques,
                                afficher_erreur, avertir, etiquette_secondaire, widget_notes)
from gui.workers import ExecuteurTache


class CarteDemo(Carte):
    """Une démonstration : bouton, progression, puis résultats (lecteurs, métriques, image)."""

    etat_modifie = Signal(bool)     # True : une démo démarre ; False : elle est terminée

    def __init__(self, titre: str, description: str, texte_bouton: str,
                 fabrique_tache: Callable[[Path], tuple],
                 construire_resultats: Callable[[ResultatTraitement, QVBoxLayout], None],
                 dossier_base: Callable[[], Path],
                 widget_options: Optional[QWidget] = None,
                 verifier: Optional[Callable[[], Optional[str]]] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(titre, parent)
        self._fabrique_tache = fabrique_tache
        self._construire_resultats = construire_resultats
        self._dossier_base = dossier_base
        self._verifier = verifier
        self._etat_fin_ok = False

        self.contenu.addWidget(etiquette_secondaire(description))
        if widget_options is not None:
            self.contenu.addWidget(widget_options)
        ligne = QHBoxLayout()
        self._btn = QPushButton(texte_bouton)
        self._btn.setProperty("primaire", True)
        self._btn.clicked.connect(self._lancer)
        self._btn_annuler = QPushButton("Annuler")
        self._btn_annuler.clicked.connect(self._annuler)
        self._btn_annuler.hide()
        ligne.addWidget(self._btn)
        ligne.addWidget(self._btn_annuler)
        ligne.addStretch(1)
        self.contenu.addLayout(ligne)
        self._barre = QProgressBar()
        self._barre.setTextVisible(False)
        self._barre.hide()
        self.contenu.addWidget(self._barre)
        self._statut = QLabel("")
        self._statut.setObjectName("Statut")
        self._statut.setWordWrap(True)
        self.contenu.addWidget(self._statut)
        self._zone = QVBoxLayout()
        self._zone.setSpacing(12)
        self.contenu.addLayout(self._zone)

        self._tache = ExecuteurTache(self)
        self._tache.statut.connect(lambda t: self._poser_statut(t, "info"))
        self._tache.progression.connect(self._sur_progression)
        self._tache.termine.connect(self._sur_termine)
        self._tache.erreur.connect(self._sur_erreur)
        self._tache.annule.connect(self._sur_annulation)
        self._tache.fini.connect(self._sur_fini)

    # ------------------------------------------------------------------
    def en_cours(self) -> bool:
        return self._tache.en_cours()

    def definir_bloque(self, bloque: bool) -> None:
        """Désactive le lancement quand l'autre démonstration tourne."""
        self._btn.setEnabled(not bloque and not self._tache.en_cours())

    def _poser_statut(self, texte: str, type_: str) -> None:
        self._statut.setText(texte)
        self._statut.setProperty("etat", type_)
        style = self._statut.style()
        style.unpolish(self._statut)
        style.polish(self._statut)

    def _lancer(self) -> None:
        if self._tache.en_cours():
            return
        probleme = self._verifier() if self._verifier else None
        if probleme:
            avertir(self, "Démonstration", probleme)
            return
        try:
            dossier = creer_dossier_sortie(self._dossier_base())
        except OSError as exc:
            avertir(self, "Dossier de sortie",
                    f"Impossible de créer le dossier de résultats :\n{exc}")
            return
        self._vider_resultats()
        self._etat_fin_ok = False
        self._btn.setEnabled(False)
        self._btn_annuler.setEnabled(True)
        self._btn_annuler.show()
        self._barre.setRange(0, 0)
        self._barre.show()
        self._poser_statut("Démarrage de la démonstration…", "info")
        self.etat_modifie.emit(True)
        fonction, args = self._fabrique_tache(dossier)
        self._tache.lancer(fonction, *args)

    def _annuler(self) -> None:
        self._btn_annuler.setEnabled(False)
        self._poser_statut("Annulation en cours (à la prochaine étape)…", "info")
        self._tache.demander_annulation()

    def _sur_progression(self, valeur: float) -> None:
        if valeur < 0:
            self._barre.setRange(0, 0)
        else:
            self._barre.setRange(0, 1000)
            self._barre.setValue(int(max(0.0, min(1.0, valeur)) * 1000))

    def _sur_termine(self, resultat: ResultatTraitement) -> None:
        self._barre.setRange(0, 1000)
        self._barre.setValue(1000)
        try:
            self._vider_resultats()
            self._construire_resultats(resultat, self._zone)
            notes = widget_notes(resultat.notes)
            if notes is not None:
                self._zone.addWidget(notes)
            fichiers = list(resultat.fichiers.values())
            self._zone.addWidget(BarreDossierSortie(resultat.dossier, fichiers))
        except Exception:
            import traceback
            self._poser_statut("La démonstration est terminée, mais l'affichage a échoué.", "erreur")
            afficher_erreur(self, "L'affichage des résultats a échoué.", traceback.format_exc())
            return
        self._etat_fin_ok = True
        self._poser_statut(f"La démonstration est terminée avec succès "
                           f"({resultat.duree_calcul_s:.1f} s de calcul).", "succes")

    def _sur_erreur(self, message: str, trace: str) -> None:
        self._poser_statut(message, "erreur")
        afficher_erreur(self, message, trace)

    def _sur_annulation(self) -> None:
        self._poser_statut("Démonstration annulée.", "info")

    def _sur_fini(self) -> None:
        self._btn.setEnabled(True)
        self._btn_annuler.hide()
        if not self._etat_fin_ok:
            self._barre.hide()
        self.etat_modifie.emit(False)

    def _vider_resultats(self) -> None:
        while self._zone.count():
            item = self._zone.takeAt(0)
            widget = item.widget()
            if widget is not None:
                for lecteur in widget.findChildren(AudioPlayerWidget):
                    lecteur.liberer()
                if isinstance(widget, AudioPlayerWidget):
                    widget.liberer()
                widget.setParent(None)
                widget.deleteLater()

    def liberer_ressources(self) -> None:
        for lecteur in self.findChildren(AudioPlayerWidget):
            lecteur.liberer()


def _ajouter_lecteurs(resultat: ResultatTraitement, disposition: QVBoxLayout,
                      elements: list[tuple[str, str]]) -> None:
    for cle, titre in elements:
        lecteur = AudioPlayerWidget(titre)
        lecteur.charger(resultat.fichiers[cle])
        disposition.addWidget(lecteur)


def _ajouter_metriques_et_image(resultat: ResultatTraitement, disposition: QVBoxLayout, titre_image: str) -> None:
    panneau = PanneauMetriques()
    panneau.definir(resultat.metriques, colonnes=2)
    disposition.addWidget(panneau)
    vue = ImageViewer(titre_image)
    vue.charger(resultat.fichiers["spectrogrammes"])
    disposition.addWidget(vue)


class PageDemo(QWidget):
    """Page des démonstrations ; même interface de cycle de vie que les autres pages."""

    retour_accueil = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("Page")
        self._dossier_sortie: Path = dossier_sortie_par_defaut()

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

        entete = QHBoxLayout()
        retour = QPushButton("← Retour")
        retour.clicked.connect(self.retour_accueil.emit)
        entete.addWidget(retour)
        entete.addStretch(1)
        colonne.addLayout(entete)
        titre = QLabel("Démonstrations")
        titre.setObjectName("TitrePage")
        colonne.addWidget(titre)
        colonne.addWidget(etiquette_secondaire(
            "Testez les algorithmes sur des signaux synthétiques : aucun fichier audio n'est nécessaire. "
            "Comme la vérité est connue, les SNR affichés sont de vraies mesures."))

        carte_dossier = Carte("Dossier de sortie")
        ligne = QHBoxLayout()
        self._label_dossier = QLabel(str(self._dossier_sortie))
        self._label_dossier.setWordWrap(True)
        changer = QPushButton("Changer…")
        changer.clicked.connect(self._choisir_dossier)
        ligne.addWidget(self._label_dossier, 1)
        ligne.addWidget(changer)
        carte_dossier.contenu.addLayout(ligne)
        colonne.addWidget(carte_dossier)

        # --- démo 1 : restauration ---
        self._carte_restauration = CarteDemo(
            "Démonstration restauration",
            "Une voix synthétique est noyée dans du vent (grondement grave) et du souffle ; "
            "le mode C (passe-haut + soustraction spectrale) la restaure.",
            "Lancer la démonstration",
            lambda dossier: (executer_demo_restauration, (dossier,)),
            lambda res, disp: (
                _ajouter_lecteurs(res, disp, [("bruite", "Signal bruité"), ("restaure", "Signal restauré")]),
                _ajouter_metriques_et_image(res, disp, "Spectrogrammes")),
            lambda: self._dossier_sortie)

        # --- démo 2 : séparation musicale ---
        options = QWidget()
        v = QVBoxLayout(options)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(QLabel("Méthode de séparation"))
        self._combo_methode = QComboBox()
        self._combo_methode.addItem("REPET-SIM (adaptée au signal synthétique)", "repet")
        self._combo_methode.addItem("NMF (expérimental)", "nmf")
        v.addWidget(self._combo_methode)
        self._carte_musique = CarteDemo(
            "Démonstration séparation musicale",
            "Une voix synthétique est mélangée à une boucle instrumentale (grosse caisse, charleston, "
            "accords) ; REPET-SIM ou NMF séparent les deux. (Demucs, conçu pour de la vraie musique, "
            "n'est pas utilisé ici : essayez-le dans le mode « Séparer une musique ».)",
            "Lancer la démonstration",
            lambda dossier: (executer_demo_musique, (self._combo_methode.currentData(), dossier)),
            lambda res, disp: (
                _ajouter_lecteurs(res, disp, [("melange", "Mélange"), ("voix", "Voix estimée"),
                                              ("instrumental", "Instrumental estimé")]),
                _ajouter_metriques_et_image(res, disp, "Spectrogrammes")),
            lambda: self._dossier_sortie, widget_options=options,
            verifier=self._verifier_methode)

        for carte, autre in ((self._carte_restauration, self._carte_musique),
                             (self._carte_musique, self._carte_restauration)):
            carte.etat_modifie.connect(lambda actif, a=autre: a.definir_bloque(actif))
            colonne.addWidget(carte)
        colonne.addStretch(1)

    def _verifier_methode(self) -> Optional[str]:
        if self._combo_methode.currentData() == "nmf" and not dependance_disponible("sklearn"):
            return "La dépendance scikit-learn est nécessaire pour utiliser NMF."
        return None

    def _choisir_dossier(self) -> None:
        dossier = QFileDialog.getExistingDirectory(self, "Choisir le dossier de sortie",
                                                   str(self._dossier_sortie))
        if dossier:
            self._dossier_sortie = Path(dossier)
            self._label_dossier.setText(dossier)

    # --- cycle de vie (même interface que PageTraitement) ---
    def _cartes(self):
        return (self._carte_restauration, self._carte_musique)

    def tache_en_cours(self) -> bool:
        return any(c.en_cours() for c in self._cartes())

    def demander_annulation(self) -> None:
        for c in self._cartes():
            c._tache.demander_annulation()

    def attendre_fin(self, delai_ms: int) -> bool:
        return all(c._tache.attendre_fin(delai_ms) for c in self._cartes())

    def liberer_ressources(self) -> None:
        for c in self._cartes():
            c.liberer_ressources()
