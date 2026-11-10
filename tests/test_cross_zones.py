"""Zonas cruzadas: cada câmera aprende as superfícies que as outras calibraram.

Caso real (P013, 09/11): olhando o notebook, a câmera da TV, que só tinha o mapa da TV,
projetava o olhar dentro da TV (x 31%, y 35%), e o painel marcava as duas telas.
"""
import math
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from modules.multicamera import Experiment, Frame
from modules.screen_gaze import ScreenGaze, camera_opinion, fuse_surfaces, zone_log_probabilities

SOURCES = {"notebook": "pc/0", "validation": "pc/1"}


def seen(camera, surface, u, v, rng):
    """O que cada câmera mede com a pessoa olhando o ponto (u, v) de uma superfície.

    Como nos dados reais (P014, P016 e P030), o que separa as telas para uma câmera é
    sobretudo a pose da cabeça (hx: +0,31 a +0,59 entre notebook e TV); a cabeça também
    acompanha um pouco cada ponto (`jitter`), o que alarga a zona de cada tela.
    """
    jitter = .1 * math.sin(7 * u + 3 * v)
    if camera == "notebook":
        base = ([.3 + .4 * u, .3 + .4 * v, .5 + jitter / 2, .5] if surface == "notebook"
                else [.62 + .05 * u, .40 + .05 * v, .15, .48])          # rosto virado para a TV
    else:
        base = ([.3 + .4 * u, .3 + .4 * v, .5 + jitter, .5] if surface == "tv"
                # Olhando o notebook, a câmera da TV vê a íris quase como olhando a TV em
                # (0,3; 0,4): o mapa da TV sozinho põe esse olhar dentro da TV, e a cabeça
                # virada para o notebook ainda cabe, na borda, na zona larga da TV.
                else [.42 + .02 * u, .46 + .02 * v, .70, .52])
    return (np.asarray(base) + rng.normal(0, .004, 4)).tolist(), .3 + rng.normal(0, .005)


class _NoNoise:
    @staticmethod
    def normal(loc, scale, size=None):
        return np.zeros(size) if size else 0.


NO_NOISE = _NoNoise()


def calibrate_world(gaze, surface, role, participant, start_at, rng, sources=SOURCES):
    gaze.start(surface, role, participant, sources[role], include_check=True, screen=surface)
    observations, displays = {}, {}
    for sequence in range(2000):
        now = start_at + sequence / 10
        cue = gaze.cue(now)
        if not cue["active"]:
            break
        for camera in sources:
            features, ear = seen(camera, surface, *cue["marker"], rng)
            observations[camera] = {"capture": now, "sequence": sequence, "features": features,
                                    "uncertainty": .01, "pose": None, "ear": ear}
        displays["window"] = {"display": surface, "visible": True, "ready": True, "cue_id": cue["cue_id"],
                              "seen": now, "layout": [1280, 720, 0, 0]}
        gaze.tick(now, observations, sources, displays)
    assert gaze.calibration["phase"] == "complete", gaze.calibration["reason"]


class CrossZoneTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.exp = Experiment(self.directory.name)
        self.exp.nodes = {"n1": {"name": "pc", "cameras": [{"id": "0"}, {"id": "1"}], "seen": time.perf_counter()}}
        self.exp.assignments = {"notebook": "n1:0", "nao": None, "validation": "n1:1"}
        rng = np.random.default_rng(3)
        calibrate_world(self.exp.screen_gaze, "notebook", "notebook", "P013", 0, rng)
        calibrate_world(self.exp.screen_gaze, "tv", "validation", "P013", 500, rng)
        self.rng = rng

    def tearDown(self):
        self.directory.cleanup()

    def look(self, surface, u, v):
        now = time.perf_counter()
        for camera in SOURCES:
            features, ear = seen(camera, surface, u, v, self.rng)
            self.exp.observations[camera] = {"capture": now, "sequence": now, "features": features,
                                             "uncertainty": .01, "ear": ear}
        data = self.exp.live_gaze(now, participant="P013")
        estimates = {s: data["screens"][s]["estimates"][0]["gaze"] for s in ("notebook", "tv")}
        return data, estimates

    def test_each_camera_learns_the_surface_the_other_calibrated(self):
        models = self.exp.screen_gaze.models_for("P013")
        self.assertEqual(set(models["notebook:notebook"]["cross_zones"]), {"validation"})
        self.assertEqual(set(models["tv:validation"]["cross_zones"]), {"notebook"})
        self.assertEqual(models["tv:validation"]["cross_zones"]["notebook"]["source"], "pc/0")

    def test_looking_at_the_notebook_does_not_light_the_tv(self):
        data, estimates = self.look("notebook", .6, .5)
        self.assertEqual(data["target"], "NOTEBOOK")
        self.assertTrue(estimates["notebook"]["inside"])
        self.assertFalse(estimates["tv"]["inside"])
        self.assertEqual(estimates["tv"]["closer_to"], "notebook")
        data, estimates = self.look("tv", .5, .5)
        self.assertEqual(data["target"], "TV")
        self.assertFalse(estimates["notebook"]["inside"])
        self.assertTrue(estimates["tv"]["inside"])

    def test_inaccurate_coordinates_do_not_cancel_recognized_screen(self):
        for surface, role in (("tv", "validation"), ("notebook", "notebook")):
            model = self.exp.screen_gaze.models_for("P013")[f"{surface}:{role}"]["map"]
            for point, visible in (([1.3, .5], True), ([3., .5], False), (None, False)):
                with self.subTest(surface=surface, point=point), patch.object(model, "predict", return_value=point):
                    data, estimates = self.look(surface, .5, .5)
                    own = estimates[surface]
                    self.assertEqual(data["target"], surface.upper())
                    self.assertTrue(own["inside"])
                    self.assertFalse(own["position_inside"])
                    self.assertEqual(own["visible"], visible)
                    if not visible:
                        self.assertNotIn("x", own)  # never invent a point at the center

    def test_overlay_recognizes_point_calibration_without_target_classifier(self):
        for surface in ("tv", "notebook"):
            self.look(surface, .5, .5)
            self.assertFalse(self.exp.classifier.models)
            for role, obs in self.exp.observations.items():
                # The eye reading accepted by point calibration can be rejected by
                # the stricter general classifier, as when looking down at a screen.
                obs.update(screen_features=obs["features"], features=None,
                           eyes=[[.4, .4], [.6, .4]], direction=[.1, 0])
                frame = Frame(None, b"", obs["capture"], obs["capture"], .01, 1)
                overlay = self.exp.eye_overlay(role, frame, "P013")
                self.assertTrue(overlay["calibrated"])
                self.assertEqual(overlay["source"], "posição")
                self.assertEqual(overlay["target"], surface.upper())
                self.assertEqual(overlay["regions"]["screens"]["target"], "DENTRO")
                self.assertNotIn("Calibre", overlay["reason"])

    def test_overlay_does_not_color_stale_or_missing_eyes(self):
        self.look("tv", .5, .5)
        obs = self.exp.observations["validation"]
        frame = Frame(None, b"", obs["capture"], obs["capture"], .01, 1)
        obs["eyes"] = None
        self.assertEqual(self.exp.eye_overlay("validation", frame, "P013")["target"], "INDETERMINADO")
        obs.update(eyes=[[.4, .4], [.6, .4]], capture=obs["capture"] - 1)
        overlay = self.exp.eye_overlay("validation", frame, "P013")
        self.assertEqual(overlay["target"], "INDETERMINADO")
        self.assertIsNone(overlay["eyes"])

    def test_preview_endpoint_uses_the_selected_participant(self):
        import json
        from server import create_app

        self.look("tv", .5, .5)
        app = create_app(self.directory.name, "test-only", start_worker=False)
        exp = app.extensions["experiment"]
        exp.screen_gaze = self.exp.screen_gaze
        exp.screen_gaze.participant = "another-participant"
        exp.nodes, exp.assignments, exp.observations = self.exp.nodes, self.exp.assignments, self.exp.observations
        obs = exp.observations["validation"]
        obs.update(eyes=[[.4, .4], [.6, .4]], direction=[.1, 0])
        exp.frames["n1:1"] = Frame(None, b"jpeg", obs["capture"], obs["capture"], .01, 1)
        client = app.test_client()
        client.post("/api/login", json={"key": "test-only"})
        response = client.get("/api/preview/validation?participant=P013")
        self.assertEqual(response.status_code, 200)
        overlay = json.loads(response.headers["X-Eye-Overlay"])
        self.assertEqual((overlay["participant"], overlay["target"]), ("P013", "TV"))
        self.assertEqual(exp.screen_gaze.participant, "another-participant")

    def test_without_cross_zones_the_tv_camera_claims_the_notebook_gaze(self):
        """O defeito da foto: cada câmera só conhece a própria tela, e as duas acendem.

        As duas câmeras estão seguras da própria tela (razões 0,26 e 0,78, longe de "fora de
        tudo"), e nenhuma conhece a tela da outra: a decisão honesta é Indeterminado.
        """
        for model in self.exp.screen_gaze.models_for("P013").values():
            model["cross_zones"] = {}
        data, estimates = self.look("notebook", .6, .5)
        self.assertTrue(estimates["notebook"]["inside"])
        self.assertTrue(estimates["tv"]["inside"])
        self.assertEqual(data["target"], "INDETERMINADO")

    def test_camera_hint_never_points_against_the_decision(self):
        """Câmeras discordando: a linha não diz "mais compatível com X" se a decisão não é X."""
        now = time.perf_counter()
        crossed = {"notebook": seen("notebook", "tv", .5, .5, NO_NOISE),        # notebook vê "TV"
                   "validation": seen("validation", "notebook", .6, .5, NO_NOISE)}  # TV vê "notebook"
        for camera, (features, ear) in crossed.items():
            self.exp.observations[camera] = {"capture": now, "sequence": now, "features": features,
                                             "uncertainty": .01, "ear": ear}
        data = self.exp.live_gaze(now, participant="P013")
        self.assertEqual(data["target"], "INDETERMINADO")
        lines = {s: data["screens"][s]["estimates"][0]["gaze"] for s in ("notebook", "tv")}
        self.assertEqual({s: (g["closer_to"], g["camera_closer_to"]) for s, g in lines.items()},
                         {"notebook": (None, "tv"), "tv": (None, "notebook")})

    def test_cross_zone_needs_the_same_camera(self):
        self.exp.nodes["n2"] = {"name": "outro", "cameras": [{"id": "0"}], "seen": time.perf_counter()}
        model = self.exp.screen_gaze.models_for("P013")["tv:validation"]
        features, ear = seen("validation", "notebook", .6, .5, self.rng)
        obs = {"capture": 1, "sequence": 1, "features": features, "uncertainty": .01, "ear": ear}
        gaze = self.exp.screen_gaze
        self.assertEqual(gaze.zone_decision("tv:validation", model, obs, "P013", SOURCES)["closer_to"], "notebook")
        # Outra câmera física no papel da TV: a zona aprendida pela pc/1 não vale para ela.
        other_camera = {**SOURCES, "validation": "outro/0"}
        self.assertIsNone(gaze.zone_decision("tv:validation", model, obs, "P013", other_camera)["closer_to"])

    def test_cross_zones_survive_restart_and_revalidation(self):
        with tempfile.TemporaryDirectory() as store:
            gaze = ScreenGaze(store)
            rng = np.random.default_rng(4)
            calibrate_world(gaze, "notebook", "notebook", "P014", 0, rng)
            reloaded = ScreenGaze(store)
            zones = reloaded.models_for("P014")["notebook:notebook"]["cross_zones"]
            self.assertEqual(set(zones), {"validation"})
            # "Validar de novo" só repete a validação; as zonas cruzadas continuam.
            reloaded.begin_check("notebook", "notebook", "P014", screen="notebook")
            observations, displays = {}, {}
            for sequence in range(2000):
                now = 900 + sequence / 10
                cue = reloaded.cue(now)
                if not cue["active"]:
                    break
                for camera in SOURCES:
                    features, ear = seen(camera, "notebook", *cue["marker"], rng)
                    observations[camera] = {"capture": now, "sequence": sequence, "features": features,
                                            "uncertainty": .01, "pose": None, "ear": ear}
                displays["w"] = {"display": "notebook", "visible": True, "ready": True, "cue_id": cue["cue_id"],
                                 "seen": now, "layout": [1280, 720, 0, 0]}
                reloaded.tick(now, observations, SOURCES, displays)
            self.assertEqual(reloaded.calibration["phase"], "complete")
            self.assertEqual(set(reloaded.models_for("P014")["notebook:notebook"]["cross_zones"]), {"validation"})


# Razões medianas reais do P014 (09/11), nos pontos de validação, fora do ajuste das zonas.
# Câmeras: notebook, validation (TV) e nao. O robô fica à frente da TV: a câmera da TV
# quase não separa os dois; a do NAO separa bem.
P014 = {
    "tv": {"notebook": {"notebook": 4.22, "tv": .62, "robot": 1.80},
           "validation": {"notebook": 3.79, "tv": .40, "robot": .27},
           "nao": {"notebook": 2.08, "tv": .60, "robot": 4.84}},
    "notebook": {"notebook": {"notebook": .58, "tv": 1.05, "robot": .86},
                 "validation": {"notebook": .53, "tv": 1.64, "robot": .48},
                 "nao": {"notebook": .49, "tv": 1.23, "robot": 1.38}},
    "robot": {"notebook": {"notebook": 3.55, "tv": 2.21, "robot": .26},
              "validation": {"notebook": 6.26, "tv": 6.95, "robot": .64},
              "nao": {"notebook": .55, "tv": 2.00, "robot": .24}},
}
OWN = {"notebook": "notebook", "validation": "tv", "nao": "robot"}


class FusionTests(unittest.TestCase):
    def test_screenshot_tv_camera_stays_on_the_tv_and_flags_the_robot(self):
        """Foto 18:16: olhando a TV, a linha dizia "fora desta tela (mais compatível com o robô)"."""
        opinion = camera_opinion("tv", P014["tv"]["validation"])
        self.assertEqual(opinion, {"in_zone": True, "closer_to": None, "ambiguous_with": "robot"})
        # A câmera do NAO separa: fora do robô, mais compatível com a TV.
        self.assertEqual(camera_opinion("robot", P014["tv"]["nao"]),
                         {"in_zone": False, "closer_to": "tv", "ambiguous_with": None})
        # Olhando o notebook, a câmera da TV não sabe se é notebook ou robô: não aponta nenhum.
        self.assertEqual(camera_opinion("tv", P014["notebook"]["validation"]),
                         {"in_zone": False, "closer_to": None, "ambiguous_with": None})

    def test_fusion_gets_each_surface_right_with_the_real_medians(self):
        for truth, evidence in P014.items():
            fused = fuse_surfaces(evidence)
            self.assertTrue(fused["decided"], truth)
            self.assertEqual(fused["surface"], truth)
        # Na votação antiga, olhando a TV, o voto da câmera da TV no robô e o da câmera do
        # NAO contra o robô se anulavam. Aqui a TV ganha com folga.
        self.assertGreater(fuse_surfaces(P014["tv"])["margin"], 3)

    def test_fusion_outside_unknown_and_disagreement(self):
        far = {"notebook": {"notebook": 3.}, "validation": {"tv": 2.5}, "nao": {"robot": 4.}}
        fused = fuse_surfaces(far)
        self.assertEqual((fused["decided"], fused["surface"]), (True, None))   # outro local
        self.assertFalse(fuse_surfaces({})["decided"])
        # Uma câmera certa do notebook e outra certa de estar fora dele: não decide.
        split = {"notebook": {"notebook": .2}, "validation": {"notebook": 3.}}
        self.assertFalse(fuse_surfaces(split)["decided"])
        # Uma câmera sem a zona do robô não é contra o robô.
        robot = {"nao": {"robot": .2}, "notebook": {"notebook": 3.}}
        self.assertEqual(fuse_surfaces(robot)["surface"], "robot")
        # Uma câmera sozinha não veta as outras duas.
        two_vs_one = {"nao": {"robot": .2, "tv": 3.}, "notebook": {"robot": .3, "notebook": 3.},
                      "validation": {"robot": 9., "tv": .2}}
        self.assertEqual(fuse_surfaces(two_vs_one)["surface"], "robot")

    def test_zone_log_probabilities_handle_far_readings(self):
        logp = zone_log_probabilities({"tv": 40., "robot": .3})
        self.assertLess(logp["tv"], -1000)
        self.assertAlmostEqual(sum(math.exp(v) for v in logp.values()), 1.)


if __name__ == "__main__":
    unittest.main()
