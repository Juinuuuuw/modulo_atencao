"""Filtro temporal das leituras ao vivo e histerese do alvo mostrado."""
import tempfile
import time
import unittest

import numpy as np

from modules.multicamera import Experiment, Frame
from modules.screen_gaze import ScreenGaze, live_view
from modules.temporal_filter import FeatureSmoother, Hysteresis, StableTarget
from modules.visual_attention_monitor import VisualObservation
from test_position_library import calibrate, vector


class FeatureSmootherTests(unittest.TestCase):
    def test_reduces_jitter_and_follows_a_gaze_shift(self):
        rng = np.random.default_rng(0)
        f = FeatureSmoother()
        fixed = np.array([.5, .45, .55, .5, .3])
        out = [f.update(i / 30, fixed + rng.normal(0, .01, 5) * [1, 1, 1, 1, 0]) for i in range(60)]
        raw_jump = np.median(np.abs(np.diff(rng.normal(0, .01, 60))))
        jump = np.median(np.abs(np.diff(np.asarray(out[10:])[:, 0])))
        self.assertLess(jump, raw_jump / 3)
        # O olhar muda de lugar: em 0,4 s o filtro já chegou perto do novo valor.
        moved = fixed + [.1, 0, 0, 0, 0]
        for i in range(60, 72):
            value = f.update(i / 30, moved)
        self.assertAlmostEqual(value[0], .6, delta=.01)

    def test_blink_frames_do_not_enter_and_long_gap_restarts(self):
        f = FeatureSmoother()
        for i in range(10):
            f.update(i / 30, [.5, .5, .5, .5, .3])
        # Na piscada a íris "pula" e a EAR cai: o valor anterior fica.
        held = f.update(10 / 30, [.9, .9, .5, .5, .1])
        np.testing.assert_allclose(held, [.5, .5, .5, .5, .3])
        self.assertIsNone(f.update(11 / 30, None))
        # Depois de 1 s sem leitura, o filtro recomeça do quadro atual, sem arrastar o antigo.
        np.testing.assert_allclose(f.update(1.5, [.2, .2, .5, .5, .3]), [.2, .2, .5, .5, .3])

    def test_live_view_keeps_raw_values_when_there_is_no_smoothing(self):
        raw = {"screen_features": [1, 2, 3, 4], "ear": .3}
        self.assertIs(live_view(raw), raw)
        view = live_view({**raw, "smooth": {"screen_features": [5, 6, 7, 8], "ear": .25}})
        self.assertEqual((view["screen_features"], view["ear"]), ([5, 6, 7, 8], .25))


class HysteresisTests(unittest.TestCase):
    def test_flicker_does_not_switch_but_a_lasting_target_does(self):
        h = Hysteresis()
        self.assertEqual(h.update(0, "TV"), "TV")
        for i in range(1, 10):  # alterna a cada 0,1 s: nunca dura 0,35 s
            self.assertEqual(h.update(i / 10, "NOTEBOOK" if i % 2 else "TV"), "TV")
        self.assertEqual(h.update(1.0, "NOTEBOOK"), "TV")
        self.assertEqual(h.update(1.2, "NOTEBOOK"), "TV")
        self.assertEqual(h.update(1.4, "NOTEBOOK"), "NOTEBOOK")

    def test_short_gaps_keep_the_target_and_long_ones_clear_it(self):
        h = Hysteresis()
        h.update(0, "NAO")
        self.assertEqual(h.update(.1, "INDETERMINADO"), "NAO")
        self.assertEqual(h.update(.6, "INDETERMINADO"), "NAO")   # piscada / discordância curta
        self.assertEqual(h.update(.95, "INDETERMINADO"), "INDETERMINADO")
        # Painel fechado por mais de 2 s: mostra a leitura atual direto.
        self.assertEqual(h.update(5, "TV"), "TV")

    def test_stable_target_reports_part_only_while_on_the_robot(self):
        s = StableTarget()
        self.assertEqual(s.update(0, "NAO", "head"), {"target": "NAO", "part": "head", "pending": None, "held_s": 0})
        state = s.update(.1, "NAO", "left-leg")
        self.assertEqual((state["part"], state["pending"]), ("head", None))
        self.assertEqual(s.update(.5, "NAO", "left-leg")["part"], "left-leg")
        state = s.update(.6, "TV")
        self.assertEqual((state["target"], state["part"], state["pending"]), ("NAO", "left-leg", "TV"))
        state = s.update(1.0, "TV")
        self.assertEqual((state["target"], state["part"]), ("TV", None))


class LiveStableTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.exp = Experiment(self.directory.name)
        self.exp.nodes = {"n1": {"name": "pc", "cameras": [{"id": "0"}, {"id": "1"}], "seen": time.perf_counter()}}
        self.exp.assignments = {"notebook": "n1:0", "nao": None, "validation": "n1:1"}

    def tearDown(self):
        self.directory.cleanup()

    def observe(self, now, points):
        for role, point in points.items():
            self.exp.observations[role] = {"capture": now, "sequence": now, "features": vector(point),
                                           "uncertainty": .01, "ear": .3}

    def test_panel_holds_target_through_a_brief_disagreement(self):
        gaze = self.exp.screen_gaze
        calibrate(gaze, surface="notebook", role="notebook", participant="P020", source="pc/0")
        calibrate(gaze, surface="notebook", role="validation", participant="P020", source="pc/1", start_at=500)
        now = time.perf_counter()
        self.observe(now, {"notebook": (.4, .5), "validation": (.45, .55)})
        self.assertEqual(self.exp.live_gaze(now)["stable"]["target"], "NOTEBOOK")
        # Uma câmera vê fora por 0,2 s: a decisão instantânea vira empate, o painel não pisca.
        self.observe(now + .1, {"notebook": (.4, .5), "validation": (.5, -2)})
        data = self.exp.live_gaze(now + .1)
        self.assertEqual((data["target"], data["stable"]["target"]), ("INDETERMINADO", "NOTEBOOK"))
        self.observe(now + .3, {"notebook": (.4, .5), "validation": (.45, .55)})
        self.assertEqual(self.exp.live_gaze(now + .3)["stable"]["target"], "NOTEBOOK")

    def test_position_uses_the_smoothed_reading_but_hides_on_a_raw_blink(self):
        gaze = self.exp.screen_gaze
        calibrate(gaze, surface="notebook", role="notebook", participant="P021", source="pc/0")
        labels = {"notebook": "pc/0"}
        now = 10.
        obs = {"capture": now, "sequence": 1, "features": vector((.9, .9)), "uncertainty": .01, "ear": .3,
               "smooth": {"screen_features": vector((.3, .4)), "ear": .3}}
        point = gaze.position("notebook:notebook", now, {"notebook": obs}, labels, {}, require_view=False)
        self.assertAlmostEqual(point["x"], .3, delta=.03)
        self.assertAlmostEqual(point["y"], .4, delta=.03)
        blink = {**obs, "ear": .05}
        self.assertFalse(gaze.position("notebook:notebook", now, {"notebook": blink}, labels, {},
                                       require_view=False)["visible"])


class FakeExtractor:
    """Íris tremendo em torno de um ponto fixo, como numa webcam real."""

    def __init__(self):
        self.rng = np.random.default_rng(1)

    def extract(self, image, timestamp):
        gx, gy = .5 + self.rng.normal(0, .02, 2)
        return VisualObservation(True, timestamp, ear=.3, gaze_feature_x=gx, gaze_feature_y=gy,
                                 iris_diameter=12, head_x=.5, head_y=.5, glabella=(320, 200))

    def close(self):
        pass


class WorkerSmoothingTests(unittest.TestCase):
    def test_worker_keeps_raw_features_and_adds_the_smoothed_ones(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = Experiment(directory, extractor_factory=FakeExtractor)
            exp.assignments = {"notebook": "n1:0", "nao": None, "validation": None}
            exp.start_worker()
            try:
                raw, smooth = [], []
                for sequence in range(40):
                    capture = time.perf_counter()
                    exp.live_viewer_seen = capture  # aba ao vivo aberta: ~15 Hz
                    with exp.lock:
                        exp.frames["n1:0"] = Frame(np.zeros((4, 4, 3), np.uint8), b"", capture, capture, .01, sequence)
                    deadline = time.perf_counter() + 1
                    while time.perf_counter() < deadline:
                        with exp.lock:
                            obs = exp.observations.get("notebook")
                        if obs and obs["sequence"] == sequence:
                            break
                        exp.stop_event.wait(.005)
                    self.assertEqual(obs["sequence"], sequence)
                    raw.append(obs["screen_features"][0])
                    smooth.append(obs["smooth"]["screen_features"][0])
                    self.assertEqual(obs["smooth"]["features"], obs["smooth"]["screen_features"])
            finally:
                exp.close()
        # Quadros chegam a ~30 Hz ou mais rápido; o suavizado treme bem menos que o bruto.
        self.assertLess(np.std(np.diff(smooth[10:])), np.std(np.diff(raw[10:])) / 2)


if __name__ == "__main__":
    unittest.main()
