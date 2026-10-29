# Audio Source Separation & Denoising Suite

Application de bureau (Windows, macOS, Linux) pour **séparer, restaurer et analyser des fichiers audio** (séparation haute qualité par **Demucs**, moteurs REPET-SIM et NMF conservés).
Interface en français (PySide6). Les algorithmes viennent du script `audio_suite_complet.py` : ils n'ont pas
été réécrits, seulement répartis en modules.

---

## 1. Installation

### 1.1 Installer Python
Python **3.10 à 3.13** (Demucs : voir 1.5 selon votre version). Téléchargement : <https://www.python.org/downloads/>.
Sous Windows, cochez **« Add python.exe to PATH »** pendant l'installation.
Vérification : `python --version` (sous macOS/Linux, essayez `python3 --version`).

### 1.2 Créer un environnement virtuel
Placez-vous dans le dossier `audio_app` (celui qui contient `main.py`) :

```
cd chemin/vers/audio_app
python -m venv .venv
```

Activation :

| Système | Commande |
|---|---|
| Windows (PowerShell) | `.venv\Scripts\Activate.ps1` |
| Windows (cmd) | `.venv\Scripts\activate.bat` |
| macOS / Linux | `source .venv/bin/activate` |

> PowerShell peut refuser le script d'activation. Dans ce cas, une seule fois :
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, puis réessayez.

### 1.3 Installer les dépendances
```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
Dépendances : PySide6, NumPy, SciPy, Matplotlib, scikit-learn (NMF).

### 1.4 Installer FFmpeg (nécessaire uniquement pour les formats autres que WAV)
FFmpeg décode MP3, M4A, FLAC, OGG, AAC, MPEG… Les fichiers **WAV** fonctionnent sans FFmpeg.

| Système | Commande |
|---|---|
| Windows | `winget install ffmpeg` (puis **fermer et rouvrir** le terminal) |
| macOS | `brew install ffmpeg` |
| Debian / Ubuntu | `sudo apt install ffmpeg` |

Vérification : `ffmpeg -version` doit afficher un numéro de version.

### 1.5 Installer Demucs (séparation haute qualité, recommandée pour les modes Musique et Podcast)

Demucs est un réseau de neurones. Il est **optionnel** : sans lui, l'application démarre normalement et les modes
REPET-SIM, NMF, Nettoyer une voix et Démonstrations fonctionnent ; si vous lancez Demucs sans l'avoir installé,
l'application affiche un message avec les commandes ci-dessous (elle ne plante pas).

**a) PyTorch — version CPU (fonctionne partout, plus lente)**
```
python -m pip install torch
```

**b) PyTorch — GPU NVIDIA (beaucoup plus rapide)** : installez PyTorch *avant* Demucs avec la commande que génère
<https://pytorch.org/get-started/locally/> pour votre système et votre version de CUDA. Elle ressemble à :
```
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```
(`cu121` est un exemple : choisissez celle proposée par le site.) Vérification :
`python -c "import torch; print(torch.cuda.is_available())"` doit afficher `True`. L'application détecte seule
le GPU et affiche le périphérique réellement utilisé ; en cas d'erreur GPU, le calcul continue sur le CPU.

**c) Demucs** — selon votre version de Python :

| Python | Commande |
|---|---|
| 3.12 ou 3.13 | `python -m pip install demucs-infer` |
| 3.11 ou moins | `python -m pip install demucs` |

Le paquet d'origine `demucs` impose d'anciennes versions de PyTorch (< 2.1), incompatibles avec Python 3.13 ;
`demucs-infer` est un fork communautaire d'inférence pour PyTorch 2.x qui expose la même API (l'application
essaie `demucs`, puis `demucs_infer`). Le fichier `requirements-demucs.txt` regroupe ces lignes.

> **Non vérifié par l'auteur de cette version** : l'installation de Demucs n'a pas pu être testée (environnement
> de développement sans accès à Internet). Si `pip` échoue, créez un environnement avec **Python 3.11**
> (`py -3.11 -m venv .venv` sous Windows) puis installez `torch` et `demucs`.

**d) Premier lancement** : le modèle (`htdemucs`, quelques centaines de Mo) est téléchargé automatiquement dans le
cache de PyTorch (`~/.cache/torch/hub` ou équivalent) : **connexion Internet requise une seule fois**. L'application
l'indique dans la barre d'état. Ensuite il est réutilisé sans nouveau téléchargement.

**e) Durée** : sur CPU, comptez plusieurs minutes pour un morceau de quelques minutes (et 4 à 8 Go de RAM) ;
`htdemucs_ft` est environ 4 fois plus lent que `htdemucs`.

---

## 2. Lancement

Depuis le dossier `audio_app`, environnement activé :

```
python main.py
```

Sous Windows, pour ne pas garder de console ouverte : `pythonw main.py`.
Aucune commande n'est ensuite nécessaire : tout se fait dans la fenêtre.
Le bouton « Dépendances » (barre latérale) vérifie ce qui manque et affiche les commandes d'installation.

---

## 3. Les quatre modes

Dans les trois modes à fichier : *Choisir un fichier audio* → régler les paramètres → lancer.
Chaque traitement crée un **sous-dossier horodaté** (`resultats_AAAA-MM-JJ_HHMM`) dans le dossier de sortie
choisi (par défaut `~/AudioSuite_resultats`) : **aucun fichier existant n'est écrasé**.
La page conserve fichier et résultats quand vous revenez à l'accueil ; « Recommencer » efface la page.

### Séparer une musique (mode A)
- **Demucs — Haute qualité (recommandé, sélectionné par défaut)** : modèle `htdemucs` (ou `htdemucs_ft`, plus lent,
  pas forcément meilleur partout). `vocals.wav` = stem *vocals* ; `instrumental.wav` = somme des stems *drums + bass + other*
  (ou le stem *no_vocals* si le modèle en fournit un). Sorties en stéréo, 44100 Hz, WAV 16 bits.
  Le calcul se fait par blocs de 30 s (recouvrement 3 s, fondu linéaire) : l'avancement affiché est réel et l'annulation
  prend effet entre deux blocs. Aucun post-traitement spectral n'est ajouté après Demucs.
- **REPET-SIM — Rapide, qualité limitée** : estime la partie qui se *répète* (batterie, basse, boucles) ; le reste est la voix.
- **NMF — Expérimental, qualité limitée** : factorisation du spectrogramme, puis classement heuristique voix/instrumental.
- Fréquence de travail (22050 Hz par défaut) : REPET-SIM et NMF seulement.
- Fichiers : `vocals.wav`, `instrumental.wav`, `spectrogrammes_musique.png`. Affichés dans les résultats : méthode, modèle,
  périphérique de calcul (CPU / nom du GPU), temps de traitement, dossier de sortie.

### Nettoyer une voix (mode C)
- Filtre passe-haut Butterworth (coupure automatique 100–120 Hz) puis soustraction spectrale.
- Paramètre **alpha** de 1,0 (douce) à 3,0 (forte, risque d'artefacts), 2,0 par défaut.
- Affiche la coupure réellement utilisée et le SNR **estimé** avant/après.
- Fichiers : `voix_restauree.wav`, `spectrogrammes_restauration.png`. Bouton « Exporter » pour en copier un exemplaire ailleurs.

### Podcast / Radio (mode B)
- Détecte, seconde par seconde, **parole** (bleu), **musique** (orange), **mixte** (violet), **silence** (gris).
- Moteur des passages mixtes : **Demucs (recommandé)** ou REPET-SIM.
- Avec Demucs : la parole seule garde son **signal original** dans la voix, la musique seule dans le fond ; seuls les passages
  mixtes sont traités (les passages mixtes séparés de moins de 2 s sont regroupés ; Demucs reçoit 2 s de contexte de chaque côté).
  Les pistes sont reconstruites par **fondus enchaînés** de 250 ms au plus, raccourcis automatiquement pour les segments courts
  (aucun trou ni doublon : la somme des poids vaut 1 à chaque échantillon). Enregistrement sans musique : la voix est
  l'original, sans séparation.
- Fichiers : `voix_podcast_claire.wav`, `musique_de_fond.wav`, `segmentation_podcast.png`, `spectrogrammes_podcast.png`
  + tableau début / fin / type.

### Démonstrations
Signaux synthétiques générés par le programme (aucun fichier requis). Comme la vérité est connue, le **SNR réel est mesuré** :
- restauration (voix + vent + souffle) ;
- séparation musicale (voix + boucle instrumentale), au choix REPET-SIM ou NMF.

---

## 4. Architecture

```
audio_app/
├── main.py                  point d'entrée
├── requirements.txt         dépendances de base
├── requirements-demucs.txt  dépendances optionnelles de Demucs (PyTorch, Demucs)
├── core/                    logique audio, SANS dépendance à Qt (testable seule)
│   ├── dsp.py               STFT, STFT inverse, masque doux
│   ├── audio_io.py          lecture (WAV direct / FFmpeg), métadonnées, écriture WAV
│   ├── restoration.py       passe-haut, profil de bruit, soustraction spectrale (mode C)
│   ├── separation.py        REPET-SIM et NMF (anciens moteurs, conservés)
│   ├── demucs_separator.py  Demucs : modèle, GPU/CPU, blocs, repli d'erreur, stems (imports PyTorch paresseux)
│   ├── podcast.py           segmentation ; isolation REPET-SIM ou Demucs (regroupement, contexte, fondus)
│   ├── metrics.py           SNR mesuré, SNR estimé, parts d'énergie
│   ├── plots.py             spectrogrammes et segmentation (PNG, thread-safe)
│   ├── pipelines.py         traitements complets des 3 modes (appelés par l'interface)
│   ├── demos.py             générateurs synthétiques + démonstrations
│   ├── environnement.py     FFmpeg, dépendances, dossiers horodatés, ouverture de dossier
│   ├── resultats.py         structures partagées (ResultatTraitement, Metrique, rappel)
│   └── erreurs.py           exceptions aux messages lisibles
├── gui/
│   ├── main_window.py       barre latérale + QStackedWidget, fermeture sécurisée
│   ├── home_page.py         accueil (4 cartes) + vérification des dépendances
│   ├── page_base.py         socle commun des pages à fichier (états, worker, résultats)
│   ├── music_page.py · restoration_page.py · podcast_page.py · demo_page.py
│   ├── audio_player_widget.py   lecteur réutilisable (QMediaPlayer)
│   ├── result_widgets.py    cartes, métriques, visionneuse d'image, tableau, dialogues
│   ├── workers.py           QThread + signaux (aucun widget touché hors thread principal)
│   └── styles.py            feuille de style (bleu #1565C0 / #1976D2)
└── tests/test_core.py · test_demucs.py   39 tests du cœur (Demucs simulé)
```

Principes : les calculs tournent dans un **QThread** ; l'interface ne reçoit que des **signaux**. Les erreurs deviennent
un message clair + une fenêtre « Détails techniques » (trace complète). La progression affichée est **réelle** : étapes
signalées par le pipeline, avancement des blocs pour REPET-SIM ; pour NMF, scikit-learn ne donne pas d'avancement, donc
la barre est **indéterminée**. Un seul lecteur audio joue à la fois.

### Ce qui a été corrigé par rapport au script d'origine (sans changer les calculs)
- `sys.exit()` dans les fonctions de lecture/séparation → **exceptions** (un `sys.exit` dans un thread le tue sans message).
- `print` et arguments de ligne de commande retirés du cœur ; la coupure du passe-haut est renvoyée via `info`.
- Matplotlib : API objet (`Figure`) au lieu de `pyplot`, qui n'est pas fait pour un thread de calcul.
- Segmentation : l'enveloppe RMS est calculée par paquets (mêmes valeurs, mais sans allouer des Go pour un long podcast).
- Séparation REPET-SIM des passages mixtes ignorée s'il n'y en a aucun (résultat identique, car pondéré par zéro).

---

## 5. Tests

```
python -m unittest discover tests -v
```
Les tests couvrent le DSP, les métriques, la segmentation, les entrées/sorties, les dossiers horodatés, les pipelines,
les démonstrations et l'annulation. `test_demucs.py` vérifie l'intégration Demucs **avec un faux modèle linéaire** (blocs,
fondus, assemblage, export, durées, repli CPU, mémoire, annulation, messages d'erreur) : il ne juge **pas** la qualité
du vrai modèle, qui ne peut se juger qu'à l'écoute. Ni Qt ni PyTorch ne sont nécessaires.

---

## 6. Limites connues

- **SNR** : sur un enregistrement réel, il est seulement *estimé* (étiqueté « estimé » dans l'interface). Il n'est *mesuré*
  que dans les démonstrations, où le signal propre est connu. Aucun SNR de séparation n'est calculable sans sources de référence.
- **Demucs** : la qualité n'a **pas pu être évaluée à l'écoute** lors du développement (PyTorch et Demucs absents, pas
  d'Internet). Il reste des fuites possibles (batterie résiduelle dans la voix, souffle de voix dans l'instrumental) ; Demucs est
  entraîné sur de la musique, pas sur de la parole : sur un podcast, sa sortie « vocals » peut altérer le timbre d'une voix
  parlée. Écoutez toujours les passages mixtes et comparez avec REPET-SIM.
- **Demucs, calcul** : plusieurs minutes sur CPU ; un bloc en cours ne s'interrompt qu'à sa fin (annulation entre deux blocs) ;
  si la fermeture ne peut pas attendre, l'application propose de forcer la fermeture. Apple MPS n'est pas utilisé (CUDA ou CPU).
  Aucune source « no_vocals » n'est inventée : l'instrumental est la somme des autres stems du modèle.
- **Mémoire insuffisante / erreur CUDA** : repli automatique sur le CPU, ou sur des segments plus courts, avec un avertissement
  dans les résultats ; si cela ne suffit pas, un message d'erreur clair est affiché.
- **REPET-SIM** suppose un instrumental qui se répète ; chœurs, voix doublées ou instruments très variables fuient dans la voix.
- **NMF** : le classement voix / instrumental est une heuristique, moins fiable que REPET-SIM.
- **Podcast** : seuils fixes et fenêtres d'une seconde ; une musique forte sous la voix peut être classée « musique » plutôt que
  « mixte ». Vérifiez à l'écoute ; seuils réglables en tête de `core/podcast.py`.
- **Restauration** : le passe-haut (100–120 Hz) coupe aussi le bas des voix très graves ; alpha élevé = artefacts « métalliques ».
- Les modes A et B travaillent à 22050 Hz par défaut (contenu utile jusqu'à 11 kHz) et en **mono**. La durée de calcul de
  REPET-SIM croît avec le carré de la durée du fichier : privilégiez des extraits de quelques minutes.
- **Lecture** : `QMediaPlayer` dépend des codecs du système. Si un format n'est pas lisible, l'application le convertit en WAV
  temporaire via FFmpeg (le fichier d'origine n'est jamais modifié) ; sans FFmpeg, l'écoute échoue mais le traitement reste possible.
  La conversion s'exécute dans le thread de l'interface : un très long fichier peut figer brièvement la fenêtre à ce moment.
- **Fermeture** pendant un calcul : l'application demande confirmation, annule à la prochaine étape et attend (jusqu'à 15 s) ;
  une étape unique très longue (ajustement NMF) ne peut pas être interrompue instantanément.
- Les métadonnées des fichiers non-WAV sont lues avec `ffprobe` (livré avec FFmpeg) ; sans lui, le fichier est décodé en entier.

## 7. Améliorations possibles
Glisser-déposer de fichiers · annulation immédiate des calculs longs · export MP3 · thèmes clair/sombre ·
modèle plus avancé (Demucs) · traitement par lots · historique des traitements · comparaison A/B avant/après.

## 8. Checklist de validation manuelle
1. `python main.py` : la fenêtre s'ouvre, la zone « Dépendances » affiche des ✓.
2. **Démonstrations** → lancer les deux démos : lecteurs, SNR « mesuré », spectrogrammes, « Ouvrir le dossier de sortie ».
3. **Nettoyer une voix** : choisir un WAV bruité, alpha 2,0, lancer ; écouter avant/après ; tester « Exporter ».
4. **Séparer une musique** : un extrait de 30 s, REPET-SIM ; écouter voix et instrumental ; essayer NMF.
5. **Podcast / Radio** : tableau de segments cohérent avec l'écoute ; couleurs identiques dans tableau, légende et graphique.
6. Choisir un MP3 (avec FFmpeg) : métadonnées affichées, traitement et lecture fonctionnent ; sans FFmpeg, message clair.
7. Choisir un fichier de moins de 5 s : message « trop court ». Lancer sans fichier : message demandant d'en choisir un.
8. Pendant un calcul : l'interface reste réactive, « Annuler » arrête à l'étape suivante ; revenir à l'accueil puis à la page
   conserve fichier et résultats ; relancer crée un **nouveau** sous-dossier horodaté.
9. Fermer la fenêtre pendant un calcul : confirmation demandée (puis proposition de forcer la fermeture si besoin).
10. **Demucs** (après installation) : mode Musique, méthode par défaut = Demucs ; sur un extrait de 30 s, la barre est
    indéterminée (un seul bloc) et la barre d'état affiche le périphérique ; `vocals.wav` et `instrumental.wav` sont
    stéréo, ont la même durée que l'original, et s'écoutent dans les lecteurs. Comparez avec REPET-SIM sur le même extrait.
11. **Demucs absent** : lancer Demucs sans l'avoir installé affiche les commandes d'installation, sans plantage.
12. **Podcast** : avec Demucs, écoutez un passage de parole seule (doit être identique à l'original) et un passage mixte.
