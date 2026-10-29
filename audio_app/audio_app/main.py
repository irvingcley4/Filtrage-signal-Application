#!/usr/bin/env python3
"""Point d'entrée : python main.py   (sous Windows, « pythonw main.py » évite la console)."""
import sys
import traceback

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow
from gui.result_widgets import afficher_erreur
from gui.styles import feuille_de_style


def _gerer_exception(type_, valeur, trace) -> None:
    """Dernier filet de sécurité : une exception non prévue s'affiche au lieu de tuer l'application."""
    texte = "".join(traceback.format_exception(type_, valeur, trace))
    sys.stderr.write(texte)
    try:
        afficher_erreur(None, "Une erreur inattendue est survenue.", texte)
    except Exception:
        pass


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Audio Source Separation & Denoising Suite")
    app.setStyleSheet(feuille_de_style())
    sys.excepthook = _gerer_exception
    fenetre = MainWindow()
    fenetre.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
