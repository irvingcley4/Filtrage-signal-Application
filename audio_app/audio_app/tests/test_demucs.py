"""Tests de l'intégration Demucs SANS PyTorch ni Demucs.

Le modèle est remplacé par un faux modèle linéaire (voix = 60 % du signal, reste = 40 %) : ces tests
vérifient tout ce qui entoure Demucs (normalisation, blocs, fondus, assemblage, exports, erreurs,
repli CPU, annulation), PAS la qualité de séparation du vrai modèle, qui ne peut se juger qu'à l'écoute.

    python -m unittest discover tests -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import demucs_separator as ds                                   # noqa: E402
from core import demos, pipelines, podcast                                # noqa: E402
from core.audio_io import charger_audio_stereo, sauvegarder_audio         # noqa: E402
from core.environnement import creer_dossier_sortie                       # noqa: E402
from core.erreurs import (AudioError, DependanceManquanteError,          # noqa: E402
                          TraitementAnnule)

FS = 44100


class FauxModele:
    sources = ["drums", "bass", "other", "vocals"]
    samplerate = FS

    def to(self, _):
        return self


def faux_bloc(modele, audio, device, shifts, overlap, segment, num_workers):
    """Voix = 60 % du bloc, batterie 10 %, basse 10 %, autres 20 % (somme = signal)."""
    return np.stack([0.1 * audio, 0.1 * audio, 0.2 * audio, 0.6 * audio])


class FauxSeparateur:
    """Remplace SeparateurDemucs dans les pipelines : même interface, aucun modèle."""
    appels = []

    def __init__(self, nom_modele="htdemucs", peripherique="auto", rappel=None, **kw):
        self.nom_modele, self.samplerate, self.modele = nom_modele, FS, None
        self.avertissements = []

    def charger(self):
        self.modele = object()

    def separer(self, audio, rappel=None):
        self.charger()                      # comme le vrai SeparateurDemucs : chargement à la demande
        FauxSeparateur.appels.append(audio.shape[1])
        return 0.6 * audio, 0.4 * audio

    def infos(self):
        return {"modele": self.nom_modele, "peripherique": "CPU (faux)", "device": "cpu"}

    def liberer(self):
        pass


class PatchDemucs(unittest.TestCase):
    """Installe un faux Demucs ; restaure l'original à la fin."""

    def setUp(self):
        self._orig = {nom: getattr(ds, nom) for nom in
                      ("_executer_bloc", "_charger_modele", "detecter_peripherique",
                       "verifier_demucs", "modele_probablement_en_cache")}
        ds._executer_bloc = faux_bloc
        ds._charger_modele = lambda nom: FauxModele()
        ds.detecter_peripherique = lambda pref="auto": ("cpu", "CPU (test)")
        ds.verifier_demucs = lambda: None
        ds.modele_probablement_en_cache = lambda: True
        self._orig_pipe = (pipelines.SeparateurDemucs, pipelines.verifier_demucs)
        pipelines.SeparateurDemucs, pipelines.verifier_demucs = FauxSeparateur, lambda: None
        FauxSeparateur.appels = []

    def tearDown(self):
        for nom, f in self._orig.items():
            setattr(ds, nom, f)
        pipelines.SeparateurDemucs, pipelines.verifier_demucs = self._orig_pipe


def signal_stereo(duree_s, fs=FS, graine=0):
    rng = np.random.default_rng(graine)
    t = np.arange(int(duree_s * fs)) / fs
    g = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * rng.standard_normal(len(t))
    d = 0.3 * np.sin(2 * np.pi * 330 * t) + 0.1 * rng.standard_normal(len(t))
    return np.stack([g, d]).astype(np.float32)


class TestBlocs(unittest.TestCase):
    def test_plan_couvre_tout_avec_recouvrement(self):
        for duree in (10, 44, 46, 100, 200.5, 61):
            n = int(duree * FS)
            blocs = ds.plan_blocs(n, FS)
            self.assertEqual(blocs[0][0], 0)
            self.assertEqual(blocs[-1][1], n)
            for (a, b), (c, d) in zip(blocs, blocs[1:]):
                self.assertGreaterEqual(b - c, int(ds.RECOUVREMENT_BLOCS_S * FS) - 1)   # recouvrement
                self.assertGreater(c, a)

    def test_poids_complementaires(self):
        n = 100 * FS
        blocs = ds.plan_blocs(n, FS)
        somme = np.zeros(n)
        for k, (d, f) in enumerate(blocs):
            fi = blocs[k - 1][1] - d if k > 0 else 0
            fo = f - blocs[k + 1][0] if k < len(blocs) - 1 else 0
            somme[d:f] += ds.poids_bloc(f - d, fi, fo)
        self.assertLess(np.max(np.abs(somme - 1.0)), 1e-5)


class TestSeparateur(PatchDemucs):
    def test_un_bloc_et_plusieurs_blocs_donnent_le_meme_resultat(self):
        court, long_ = signal_stereo(20), signal_stereo(100)
        for audio in (court, long_):
            progressions = []
            sep = ds.SeparateurDemucs()
            voix, instru = sep.separer(audio, rappel=lambda s, p: progressions.append(p))
            moyenne = float(audio.mean())
            self.assertEqual(voix.shape, audio.shape)
            self.assertLess(np.max(np.abs(voix - 0.6 * (audio - moyenne))), 1e-3)
            self.assertLess(np.max(np.abs(instru - 0.4 * (audio - moyenne))), 1e-3)
        reelles = [p for p in progressions if p is not None]
        self.assertGreaterEqual(len(reelles), 3)                       # progression par blocs, réelle
        self.assertEqual(reelles[:-1], sorted(reelles[:-1]))

    def test_annulation_entre_deux_blocs(self):
        compte = []

        def rappel(statut, p):
            compte.append(statut)
            if len(compte) >= 2:
                raise TraitementAnnule()

        with self.assertRaises(TraitementAnnule):
            ds.SeparateurDemucs().separer(signal_stereo(100), rappel=rappel)

    def test_repli_cpu_apres_erreur_cuda(self):
        ds.detecter_peripherique = lambda pref="auto": ("cuda", "GPU NVIDIA (test)")

        def bloc(modele, audio, device, *a):
            if device == "cuda":
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return faux_bloc(modele, audio, device, *a)

        ds._executer_bloc = bloc
        sep = ds.SeparateurDemucs()
        voix, _ = sep.separer(signal_stereo(10))
        self.assertEqual(sep.device, "cpu")
        self.assertTrue(sep.avertissements)
        self.assertTrue(np.all(np.isfinite(voix)))

    def test_memoire_cpu_segment_reduit_puis_echec(self):
        appels = []

        def bloc(modele, audio, device, shifts, overlap, segment, nw):
            appels.append(segment)
            if segment is None:
                raise MemoryError()
            return faux_bloc(modele, audio, device, shifts, overlap, segment, nw)

        ds._executer_bloc = bloc
        ds.SeparateurDemucs().separer(signal_stereo(10))
        self.assertEqual(appels, [None, ds.SEGMENT_REDUIT_S])

        def toujours(*a):
            raise MemoryError()

        ds._executer_bloc = toujours
        with self.assertRaises(AudioError) as ctx:
            ds.SeparateurDemucs().separer(signal_stereo(10))
        self.assertIn("Mémoire insuffisante", str(ctx.exception))

    def test_erreurs_de_signal_et_de_modele(self):
        with self.assertRaises(AudioError):                              # silence
            ds.SeparateurDemucs().separer(np.zeros((2, FS * 6), dtype=np.float32))
        with self.assertRaises(AudioError):                              # mono
            ds.SeparateurDemucs().separer(np.ones((1, FS)))
        with self.assertRaises(AudioError):                              # valeurs invalides
            mauvais = signal_stereo(6); mauvais[0, 10] = np.nan
            ds.SeparateurDemucs().separer(mauvais)

        class SansVoix(FauxModele):
            sources = ["drums", "bass", "other"]

        ds._charger_modele = lambda nom: SansVoix()
        with self.assertRaises(AudioError) as ctx:
            ds.SeparateurDemucs().charger()
        self.assertIn("vocals", str(ctx.exception))

    def test_telechargement_impossible_donne_un_message_clair(self):
        def echec(nom):
            raise OSError("<urlopen error [Errno -2] Name or service not known>")

        ds._charger_modele = echec
        with self.assertRaises(AudioError) as ctx:
            ds.SeparateurDemucs().charger()
        self.assertIn("Internet", str(ctx.exception))

    def test_stem_no_vocals_utilise_quand_il_existe(self):
        class AvecNoVocals(FauxModele):
            sources = ["no_vocals", "vocals"]

        ds._charger_modele = lambda nom: AvecNoVocals()
        ds._executer_bloc = lambda m, a, *r: np.stack([0.4 * a, 0.6 * a])
        voix, instru = ds.SeparateurDemucs().separer(signal_stereo(6))
        self.assertLess(np.max(np.abs(voix + instru - (signal_stereo(6) - signal_stereo(6).mean()))), 1e-3)

    def test_validation_des_sorties(self):
        a = np.ones((2, 100), dtype=np.float32)
        ds.valider_sorties(a, a, 100)
        with self.assertRaises(AudioError):
            ds.valider_sorties(a, a[:, :90], 100)
        with self.assertRaises(AudioError):
            ds.valider_sorties(a * np.nan, a, 100)
        with self.assertRaises(AudioError):
            ds.valider_sorties(a[:, :0], a[:, :0], 0)

    def test_demucs_absent_message_avec_commandes(self):
        for nom, f in self._orig.items():
            setattr(ds, nom, f)
        original = ds.demucs_disponible
        ds.demucs_disponible = lambda: False
        try:
            with self.assertRaises(DependanceManquanteError) as ctx:
                ds.verifier_demucs()
            self.assertIn("pip install", str(ctx.exception))
        finally:
            ds.demucs_disponible = original


class TestPodcastDemucs(PatchDemucs):
    def test_regroupement_des_mixtes_proches(self):
        self.assertEqual(podcast.regrouper_mixtes(["mixte", "parole", "mixte", "parole", "parole", "parole", "mixte"]),
                         ["mixte", "mixte", "mixte", "parole", "parole", "parole", "mixte"])

    def test_ponderations_somment_a_un_et_fondus_adaptatifs(self):
        n = 10 * FS
        for durees in ([1, 1, 1, 7], [3, 0.3, 2, 4.7], [5, 5]):
            bornes, a = [], 0
            for d in durees:
                b = min(n, a + int(round(d * FS)))
                bornes.append((a, b)); a = b
            bornes[-1] = (bornes[-1][0], n)
            somme = np.zeros(n)
            for i0, i1, w in podcast.calculer_ponderations(bornes, FS, n):
                somme[i0:i1] += w
            self.assertLess(np.max(np.abs(somme - 1.0)), 1e-5, durees)

    def test_assemblage_conserve_signal_et_routage(self):
        fs = 22050
        x = signal_stereo(30, fs)
        etiq = ["parole"] * 10 + ["musique"] * 10 + ["mixte"] * 10
        bornes, types = podcast.bornes_en_echantillons(podcast.fusionner_segments(etiq), fs, x.shape[1])
        a, b = bornes[2]
        c0 = a - int(2 * fs)
        sorties = {2: (c0, 0.5 * x[:, c0:], 0.5 * x[:, c0:])}
        voix, fond = podcast.assembler_pistes(x, fs, bornes, types, sorties)
        self.assertLess(np.max(np.abs(voix + fond - x)), 1e-4)             # aucun trou ni doublon
        m1, m2, m3 = 5 * fs, 15 * fs, 25 * fs                               # milieux des 3 segments
        self.assertLess(np.max(np.abs(voix[:, m1 - 100:m1 + 100] - x[:, m1 - 100:m1 + 100])), 1e-5)
        self.assertLess(np.max(np.abs(fond[:, m1 - 100:m1 + 100])), 1e-5)   # parole : rien dans le fond
        self.assertLess(np.max(np.abs(fond[:, m2 - 100:m2 + 100] - x[:, m2 - 100:m2 + 100])), 1e-5)
        self.assertLess(np.max(np.abs(voix[:, m2 - 100:m2 + 100])), 1e-5)   # musique : rien dans la voix
        self.assertLess(np.max(np.abs(voix[:, m3 - 100:m3 + 100] - 0.5 * x[:, m3 - 100:m3 + 100])), 1e-5)

    def test_isolation_complete_avec_faux_separateur(self):
        fs = 22050
        mono = np.zeros(30 * fs)
        t = np.arange(30 * fs) / fs
        voix = demos.voix_synthetique(fs, 30, lambda u: 150 + 30 * np.sin(2 * np.pi * 0.3 * u))
        musique = 0.04 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 277 * t) + np.sin(2 * np.pi * 330 * t))
        mono[:10 * fs] = voix[:10 * fs]
        mono[10 * fs:20 * fs] = 3 * musique[10 * fs:20 * fs]
        mono[20 * fs:] = voix[20 * fs:] + musique[20 * fs:]
        x = np.stack([mono, mono]).astype(np.float32)
        infos = {}
        v, f, etiq, _, _ = podcast.isoler_voix_tapis_demucs(x, fs, FauxSeparateur(), infos=infos)
        self.assertEqual(infos["regions"], 1)
        self.assertEqual(v.shape, x.shape)
        self.assertEqual(FauxSeparateur.appels, [int(12 * fs)])               # passage mixte + 2 s de contexte
        self.assertLess(np.max(np.abs(v + f - x)), 1e-4)

    def test_parole_seule_conserve_l_original(self):
        fs = 22050
        mono = demos.voix_synthetique(fs, 12, lambda u: 150 + 30 * np.sin(2 * np.pi * 0.3 * u)).astype(np.float32)
        x = np.stack([mono, mono])
        infos = {}
        v, f, etiq, _, _ = podcast.isoler_voix_tapis_demucs(x, fs, FauxSeparateur(), infos=infos)
        self.assertEqual(infos["mode"], "parole_seule")
        self.assertTrue(np.array_equal(v, x))
        self.assertFalse(np.any(f))
        self.assertEqual(FauxSeparateur.appels, [])


class TestPipelinesDemucs(PatchDemucs):
    def test_charger_audio_stereo(self):
        with tempfile.TemporaryDirectory() as tmp:
            mono = Path(tmp) / "mono.wav"
            sauvegarder_audio(mono, signal_stereo(3, 22050)[0], 22050)
            a, fs = charger_audio_stereo(mono, FS)
            self.assertEqual((a.shape[0], fs), (2, FS))
            self.assertAlmostEqual(a.shape[1] / FS, 3.0, places=2)
            self.assertTrue(np.array_equal(a[0], a[1]))
            st = Path(tmp) / "st.wav"
            sauvegarder_audio(st, signal_stereo(3, FS).T, FS)
            b, _ = charger_audio_stereo(st, FS)
            self.assertFalse(np.array_equal(b[0], b[1]))

    def test_traiter_musique_demucs_exporte_les_bons_fichiers(self):
        import wave
        statuts = []
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "chanson.wav"
            sauvegarder_audio(src, signal_stereo(8).T, FS)
            sortie = creer_dossier_sortie(Path(tmp) / "out")
            res = pipelines.traiter_musique(src, "demucs", 22050, sortie, lambda s, p: statuts.append(s))
            self.assertEqual(res.fichiers["voix"].name, "vocals.wav")
            self.assertEqual(res.fichiers["instrumental"].name, "instrumental.wav")
            self.assertTrue(res.fichiers["spectrogrammes"].name == "spectrogrammes_musique.png")
            tailles = []
            for cle in ("voix", "instrumental"):
                with wave.open(str(res.fichiers[cle])) as w:
                    tailles.append((w.getnchannels(), w.getframerate(), w.getnframes()))
            self.assertEqual(tailles[0], tailles[1])
            self.assertEqual(tailles[0], (2, FS, int(8 * FS)))
            libelles = {m.libelle for m in res.metriques}
            self.assertTrue({"Méthode", "Modèle", "Périphérique de calcul", "Temps de traitement"} <= libelles)
            self.assertTrue(all(m.nature == "info" for m in res.metriques))      # aucun faux SNR
            self.assertIn("Préparation du fichier audio…", statuts)
            self.assertIn("Export des résultats…", statuts)
            self.assertEqual(sorted(p.name for p in sortie.iterdir() if p.suffix == ".wav"),
                             ["instrumental.wav", "vocals.wav"])

    def test_musique_trop_courte(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "court.wav"
            sauvegarder_audio(src, signal_stereo(2).T, FS)
            with self.assertRaises(AudioError) as ctx:
                pipelines.traiter_musique(src, "demucs", 22050, tmp)
            self.assertIn("trop court", str(ctx.exception))

    def test_traiter_podcast_demucs(self):
        fs = 22050
        t = np.arange(30 * fs) / fs
        voix = demos.voix_synthetique(fs, 30, lambda u: 150 + 30 * np.sin(2 * np.pi * 0.3 * u))
        musique = 0.04 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 277 * t) + np.sin(2 * np.pi * 330 * t))
        mono = np.zeros(30 * fs)
        mono[:10 * fs] = voix[:10 * fs]
        mono[10 * fs:20 * fs] = 3 * musique[10 * fs:20 * fs]
        mono[20 * fs:] = voix[20 * fs:] + musique[20 * fs:]
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "pod.wav"
            sauvegarder_audio(src, mono, fs)
            res = pipelines.traiter_podcast(src, fs, creer_dossier_sortie(Path(tmp) / "out"))
            noms = sorted(p.name for p in res.dossier.iterdir())
            self.assertEqual(noms, ["musique_de_fond.wav", "segmentation_podcast.png",
                                    "spectrogrammes_podcast.png", "voix_podcast_claire.wav"])
            self.assertEqual([s[2] for s in res.segments], ["parole", "musique", "mixte"])
            self.assertTrue(any(m.libelle == "Modèle" for m in res.metriques))

    def test_moteur_inconnu(self):
        with self.assertRaises(AudioError):
            pipelines.traiter_podcast("x.wav", 22050, ".", moteur="inconnu")


if __name__ == "__main__":
    unittest.main()
