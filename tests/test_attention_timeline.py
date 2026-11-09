import unittest

from modules.attention_timeline import (
    AttentionConfig,
    AttentionTimeline,
    EXTERNAL,
    FOCUS_BY_TARGET,
    INDETERMINATE,
    SLEEPY,
)


class AttentionTimelineTests(unittest.TestCase):
    def test_external_dispersion_requires_two_seconds_and_starts_retroactively(self):
        timeline = AttentionTimeline(roles=("notebook",))
        timeline.update(0.0, "TV", tracking={"notebook": True}, supporting_cameras=("notebook",))
        closed, live = timeline.update(1.0, "OUTRO_LOCAL", tracking={"notebook": True})
        self.assertEqual(live["state"], FOCUS_BY_TARGET["TV"])
        self.assertEqual(closed, [])
        closed, live = timeline.update(3.0, "OUTRO_LOCAL", tracking={"notebook": True})
        self.assertEqual(live["state"], EXTERNAL)
        self.assertEqual(live["since"], 1.0)
        self.assertEqual(closed[0]["end"], 1.0)

    def test_short_external_glance_does_not_become_dispersion(self):
        timeline = AttentionTimeline()
        timeline.update(0.0, "TV")
        timeline.update(1.0, "OUTRO_LOCAL")
        closed, live = timeline.update(2.5, "NAO")
        self.assertEqual(live["state"], FOCUS_BY_TARGET["NAO"])
        self.assertEqual(closed[0]["state"], FOCUS_BY_TARGET["TV"])
        self.assertEqual(closed[0]["end"], 2.5)

    def test_personalized_ear_threshold_and_prolonged_closure(self):
        timeline = AttentionTimeline(ear_reference=0.30)
        self.assertAlmostEqual(timeline.ear_threshold, 0.18)
        timeline.update(0.0, "NOTEBOOK", ear=0.3)
        timeline.update(1.0, "NOTEBOOK", ear=0.15)
        closed, live = timeline.update(3.0, "NOTEBOOK", ear=0.15)
        self.assertEqual(live["state"], SLEEPY)
        self.assertEqual(live["since"], 1.0)
        self.assertEqual(closed[0]["end"], 1.0)

    def test_gaze_activity_is_recorded_without_inventing_a_cognitive_label(self):
        config = AttentionConfig(gaze_activity_window_seconds=1.0, gaze_activity_fps=3.0)
        timeline = AttentionTimeline(config)
        live = None
        for index in range(4):
            _, live = timeline.update(index / 3, "TV", gaze_point=(100.0, 100.0))
        self.assertEqual(live["gaze_activity"], 0.0)
        self.assertEqual(live["state"], FOCUS_BY_TARGET["TV"])
        self.assertIsNone(live["gaze_activity_threshold"])

    def test_data_loss_is_reported_per_camera(self):
        timeline = AttentionTimeline(roles=("notebook", "nao"))
        timeline.update(0.0, "TV", tracking={"notebook": True, "nao": False})
        timeline.update(1.0, "TV", tracking={"notebook": True, "nao": True})
        event = timeline.close(2.0)[0]
        self.assertEqual(event["data_loss_percent"]["notebook"], 0)
        self.assertEqual(event["data_loss_percent"]["nao"], 50)


if __name__ == "__main__":
    unittest.main()
