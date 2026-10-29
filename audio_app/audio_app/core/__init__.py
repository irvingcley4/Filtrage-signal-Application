"""Cœur de traitement audio (sans aucune dépendance à l'interface graphique).

Les algorithmes viennent du script `audio_suite_complet.py` : ils sont conservés
tels quels, seulement répartis en modules et débarrassés de `print`, `sys.exit`
et des arguments de ligne de commande, pour pouvoir être appelés depuis une
interface graphique ou depuis des tests.
"""

__version__ = "1.1.0"
