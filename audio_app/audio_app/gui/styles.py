"""Identité visuelle : couleurs et feuille de style Qt (QSS) de l'application."""

BLEU = "#1565C0"          # bleu principal (barre latérale, survols)
BLEU_CLAIR = "#1976D2"    # bleu des boutons principaux
FOND = "#F4F6FA"          # gris très clair des pages
TEXTE = "#1F2933"
TEXTE_SECONDAIRE = "#52606D"
BORDURE = "#E1E6EF"
SUCCES = "#2E7D32"
ERREUR = "#C62828"
ESTIME = "#E65100"


def feuille_de_style() -> str:
    return f"""
* {{ font-family: "Segoe UI", "Inter", "Helvetica Neue", "Noto Sans", Arial, sans-serif;
     font-size: 14px; color: {TEXTE}; }}
QMainWindow {{ background: {FOND}; }}
QWidget#Page, QWidget#Contenu {{ background: {FOND}; }}
QScrollArea {{ border: none; background: transparent; }}

QFrame#Carte {{ background: white; border: 1px solid {BORDURE}; border-radius: 12px; }}
QFrame#CarteMode {{ background: white; border: 1px solid {BORDURE}; border-radius: 14px; }}
QFrame#CarteMode:hover {{ border: 1px solid {BLEU_CLAIR}; }}
QFrame#LecteurAudio {{ background: #F7F9FC; border: 1px solid {BORDURE}; border-radius: 10px; }}
QFrame#Metrique {{ background: #F7F9FC; border: 1px solid {BORDURE}; border-radius: 10px; }}

QLabel#TitreApp {{ font-size: 26px; font-weight: 700; color: {BLEU}; }}
QLabel#TitrePage {{ font-size: 22px; font-weight: 700; color: {BLEU}; }}
QLabel#TitreCarte {{ font-size: 16px; font-weight: 600; }}
QLabel#TitreMode {{ font-size: 18px; font-weight: 600; }}
QLabel#TexteSecondaire {{ color: {TEXTE_SECONDAIRE}; }}
QLabel#Badge {{ background: #E3F0FD; border-radius: 24px; font-size: 22px; }}
QLabel#ValeurMetrique {{ font-size: 18px; font-weight: 700; }}
QLabel#LibelleMetrique {{ color: {TEXTE_SECONDAIRE}; font-size: 12px; }}
QLabel#TagMesure {{ background: #E8F5E9; color: {SUCCES}; border-radius: 8px; padding: 1px 8px; font-size: 11px; }}
QLabel#TagEstime {{ background: #FFF3E0; color: {ESTIME}; border-radius: 8px; padding: 1px 8px; font-size: 11px; }}
QLabel#TagInfo {{ background: #ECEFF4; color: {TEXTE_SECONDAIRE}; border-radius: 8px; padding: 1px 8px; font-size: 11px; }}
QLabel#Statut[etat="info"] {{ color: {TEXTE_SECONDAIRE}; }}
QLabel#Statut[etat="succes"] {{ color: {SUCCES}; font-weight: 600; }}
QLabel#Statut[etat="erreur"] {{ color: {ERREUR}; font-weight: 600; }}
QLabel#ErreurLecteur {{ color: {ERREUR}; font-size: 12px; }}

QPushButton {{ background: #EAF2FC; color: {BLEU}; border: 1px solid {BLEU_CLAIR};
              border-radius: 8px; padding: 8px 16px; }}
QPushButton:hover {{ background: #DCEBFA; }}
QPushButton:pressed {{ background: #CFE2F8; }}
QPushButton:disabled {{ color: #9AA5B1; border-color: #CBD2D9; background: #F0F2F5; }}
QPushButton[primaire="true"] {{ background: {BLEU_CLAIR}; color: white; border: none;
              font-weight: 600; padding: 10px 22px; }}
QPushButton[primaire="true"]:hover {{ background: {BLEU}; }}
QPushButton[primaire="true"]:disabled {{ background: #9BBCE3; color: white; }}

QFrame#Sidebar {{ background: {BLEU}; }}
QLabel#TitreSidebar {{ color: white; font-size: 15px; font-weight: 700; }}
QLabel#SousTitreSidebar {{ color: #BBD6F5; font-size: 12px; }}
QPushButton#NavBouton {{ color: white; background: transparent; text-align: left;
              padding: 11px 16px; border: none; border-radius: 8px; }}
QPushButton#NavBouton:hover {{ background: rgba(255,255,255,0.14); }}
QPushButton#NavBouton:checked {{ background: rgba(255,255,255,0.26); font-weight: 600; }}

QProgressBar {{ border: none; background: {BORDURE}; border-radius: 6px; max-height: 12px; min-height: 12px; }}
QProgressBar::chunk {{ background: {BLEU_CLAIR}; border-radius: 6px; }}

QSlider::groove:horizontal {{ height: 6px; background: {BORDURE}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {BLEU_CLAIR}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: white; border: 2px solid {BLEU_CLAIR};
              width: 14px; margin: -6px 0; border-radius: 8px; }}

QComboBox, QSpinBox, QDoubleSpinBox {{ border: 1px solid #CBD2D9; border-radius: 6px;
              padding: 5px 8px; background: white; min-height: 22px; }}
QTableWidget {{ background: white; border: 1px solid {BORDURE}; border-radius: 8px;
              gridline-color: {BORDURE}; }}
QHeaderView::section {{ background: #F0F4FA; border: none; border-bottom: 1px solid {BORDURE};
              padding: 6px; font-weight: 600; }}
QPlainTextEdit {{ background: #F7F9FC; border: 1px solid {BORDURE}; border-radius: 8px;
              font-family: "Consolas", "Menlo", "DejaVu Sans Mono", monospace; font-size: 12px; }}
"""
