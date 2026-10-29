"""Tests du cœur de traitement (sans interface graphique).

Lancer depuis le dossier audio_app :
    python -m unittest discover tests -v
"""
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import demos, pipelines                                      # noqa: E402
from core.audio_io import (charger_audio, lire_infos, sauvegarder_audio)  # noqa: E402
from core.dsp import calculer_stft, inverser_stft                       # noqa: E402
from core.environnement import creer_dossier_sortie                     # noqa: E402
from core.erreurs import (AudioError, AudioTropCourtError,              # noqa: E402
                          FichierInvalideError, TraitementAnnule)
from core.metrics import snr_db, snr_estime                             # noqa: E402
from core.podcast import (fusionner_segments, segmenter_parole_musique,  # noqa: E402
                          segments_pour_affichage)
from core.restoration import choisir_coupure_dynamique, restaurer_voix  # noqa: E402
from core.separation import separer_voix_fond                           # noqa: E402


def signal_podcast(fs=22050):
    """30 s : 0-10 parole, 10-20 musique seule, 20-30 parole + musique discrète."""
    t = np.arange(30 * fs) / fs
    voix = demos.voix_synthetique(fs, 30, lambda u: 150 + 30 * np.sin(2 * np.pi * 0.3 * u))
    musique = 0.04 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 277 * t)
                      + np.sin(2 * np.pi * 330 * t)) * (1 + 0.3 * np.sin(2 * np.pi * 2 * t))
    x = np.zeros(30 * fs)
    x[:10 * fs] = voix[:10 * fs]
    x[10 * fs:20 * fs] = 3 * musique[10 * fs:20 * fs]
    x[20 * fs:] = voix[20 * fs:] + musique[20 * fs:]
    return x, fs


class TestDsp(unittest.TestCase):
    def test_stft_reconstruction(self):
        x = np.random.default_rng(0).standard_normal(16000)
        Z = calculer_stft(x, 16000, 1024, 256)
        y = inverser_stft(Z, 16000, 1024, 256, len(x))
        self.assertEqual(len(y), len(x))
        self.assertLess(np.max(np.abs(x - y)), 1e-8)

    def test_snr_valeurs_connues(self):
        x = np.random.default_rng(1).standard_normal(5000)
        self.assertAlmostEqual(snr_db(x, 0.5 * x), 6.02, places=1)   # erreur = x/2 -> 4x moins d'énergie
        self.assertGreater(snr_db(x, x), 90)                          # estimation parfaite

    def test_snr_estime_distingue_propre_et_bruite(self):
        propre, bruite, fs = demos.generer_demo_restauration()
        self.assertGreater(snr_estime(propre, fs), snr_estime(bruite, fs))


class TestRestauration(unittest.TestCase):
    def test_coupure_dans_les_bornes(self):
        _, bruite, fs = demos.generer_demo_restauration()
        self.assertTrue(100.0 <= choisir_coupure_dynamique(bruite, fs) <= 120.0)

    def test_restauration_ameliore_le_snr_reel(self):
        propre, bruite, fs = demos.generer_demo_restauration()
        info = {}
        restaure = restaurer_voix(bruite, fs, info=info)
        self.assertIn("coupure_hz", info)
        self.assertEqual(len(restaure), len(bruite))
        self.assertGreater(snr_db(propre, restaure), snr_db(propre, bruite) + 2.0)


class TestSeparation(unittest.TestCase):
    def test_repet_ameliore_voix_et_instrumental(self):
        voix, instru, melange, fs = demos.generer_demo_musique()
        v, f = separer_voix_fond(melange, fs, "repet")
        self.assertGreater(snr_db(voix, v), snr_db(voix, melange) + 3.0)
        self.assertGreater(snr_db(instru, f), snr_db(instru, melange) + 3.0)

    def test_progression_reelle_croissante_jusqu_a_un(self):
        _, _, melange, fs = demos.generer_demo_musique()
        valeurs = []
        separer_voix_fond(melange, fs, "repet", rappel_progression=valeurs.append)
        self.assertTrue(len(valeurs) > 1)
        self.assertEqual(valeurs, sorted(valeurs))
        self.assertAlmostEqual(valeurs[-1], 1.0)


class TestPodcast(unittest.TestCase):
    def test_segmentation_trois_types(self):
        x, fs = signal_podcast()
        etiquettes, _, _ = segmenter_parole_musique(x, fs)
        self.assertEqual(etiquettes[5], "parole")
        self.assertEqual(etiquettes[15], "musique")
        self.assertEqual(etiquettes[25], "mixte")

    def test_fusion_et_bornes(self):
        self.assertEqual(fusionner_segments(["a", "a", "b"]), [(0.0, 2.0, "a"), (2.0, 3.0, "b")])
        # durée réelle 2,4 s : la 3e fenêtre est ignorée, la 2e est bornée
        self.assertEqual(segments_pour_affichage(["a", "a", "b"], 1.5), [(0.0, 1.5, "a")])


class TestEntreesSorties(unittest.TestCase):
    def test_aller_retour_wav_et_infos(self):
        x = 0.5 * np.sin(2 * np.pi * 440 * np.arange(16000 * 3) / 16000)
        with tempfile.TemporaryDirectory() as tmp:
            chemin = Path(tmp) / "a.wav"
            sauvegarder_audio(chemin, x, 16000)
            y, fs = charger_audio(chemin)
            self.assertEqual(fs, 16000)
            self.assertLess(np.max(np.abs(x - y[:len(x)])), 1e-3)
            z, fs2 = charger_audio(chemin, sr_cible=8000)
            self.assertEqual(fs2, 8000)
            self.assertAlmostEqual(len(z) / fs2, 3.0, places=2)
            infos = lire_infos(chemin)
            self.assertEqual((infos.frequence, round(infos.duree_s)), (16000, 3))

    def test_fichiers_invalides(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FichierInvalideError):
                lire_infos(Path(tmp) / "absent.wav")
            vide = Path(tmp) / "vide.wav"
            vide.write_bytes(b"")
            with self.assertRaises(FichierInvalideError):
                lire_infos(vide)

    def test_dossier_sortie_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            maintenant = datetime(2026, 10, 12, 23, 50)
            a = creer_dossier_sortie(tmp, maintenant)
            b = creer_dossier_sortie(tmp, maintenant)
            self.assertNotEqual(a, b)
            self.assertEqual(a.name, "resultats_2026-10-12_2350")
            self.assertTrue(a.is_dir() and b.is_dir())


class TestPipelines(unittest.TestCase):
    def test_fichier_trop_court(self):
        x = 0.1 * np.random.default_rng(0).standard_normal(22050)   # 1 s
        with tempfile.TemporaryDirectory() as tmp:
            chemin = Path(tmp) / "court.wav"
            sauvegarder_audio(chemin, x, 22050)
            with self.assertRaises(AudioTropCourtError):
                pipelines.traiter_musique(chemin, "repet", 22050, tmp)

    def test_alpha_invalide(self):
        with self.assertRaises(AudioError):
            pipelines.traiter_restauration("peu_importe.wav", 5.0, ".")

    def test_restauration_complete(self):
        _, bruite, fs = demos.generer_demo_restauration()
        statuts = []
        with tempfile.TemporaryDirectory() as tmp:
            chemin = Path(tmp) / "bruit.wav"
            sauvegarder_audio(chemin, bruite, fs)
            sortie = creer_dossier_sortie(Path(tmp) / "out")
            res = pipelines.traiter_restauration(chemin, 2.0, sortie, lambda s, p: statuts.append((s, p)))
            for cle in ("restaure", "spectrogrammes"):
                self.assertTrue(res.fichiers[cle].is_file())
            self.assertTrue(any(m.nature == "estime" for m in res.metriques))
            self.assertEqual(statuts[-1][1], 1.0)

    def test_podcast_complet(self):
        x, fs = signal_podcast()
        with tempfile.TemporaryDirectory() as tmp:
            chemin = Path(tmp) / "pod.wav"
            sauvegarder_audio(chemin, x, fs)
            sortie = creer_dossier_sortie(Path(tmp) / "out")
            res = pipelines.traiter_podcast(chemin, fs, sortie, moteur="repet")
            for cle in ("voix", "fond", "segmentation", "spectrogrammes"):
                self.assertTrue(res.fichiers[cle].is_file(), cle)
            self.assertEqual([t for _, _, t in res.segments], ["parole", "musique", "mixte"])

    def test_annulation_cooperative(self):
        def rappel(statut, progression):
            if progression is not None and progression > 0.2:
                raise TraitementAnnule()

        x, fs = signal_podcast()
        with tempfile.TemporaryDirectory() as tmp:
            chemin = Path(tmp) / "m.wav"
            sauvegarder_audio(chemin, x, fs)
            with self.assertRaises(TraitementAnnule):
                pipelines.traiter_musique(chemin, "repet", fs, tmp, rappel)

    def test_demos(self):
        with tempfile.TemporaryDirectory() as tmp:
            r1 = demos.executer_demo_restauration(creer_dossier_sortie(tmp))
            self.assertTrue(all(m.nature in ("mesure", "info") for m in r1.metriques))
            r2 = demos.executer_demo_musique("repet", creer_dossier_sortie(tmp))
            self.assertTrue(r2.fichiers["voix"].is_file())


if __name__ == "__main__":
    unittest.main()
