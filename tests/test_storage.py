import json
from pathlib import Path
import tempfile
import unittest

from modules.storage import AnalysisStore


class AnalysisStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.session_id = "a" * 32
        self.session = self.root / self.session_id
        self.session.mkdir()
        self.write_json("session.json", {
            "id": self.session_id, "participant": "P001",
            "started_utc": "2026-01-02T12:00:00+00:00", "duration": 2.0,
            "status": "complete", "pilot_override": ""})
        self.write_json("annotations.json", {
            "A01": [{"start": 0, "end": 2, "target": "TV"}]})
        self.write_json("report.json", {
            "annotator": "A01", "accuracy": 0.75, "coverage": 0.8,
            "macro_f1": 0.7, "kappa": 0.6})
        self.write_json("reports.json", {
            "A02": {"annotator": "A02", "accuracy": 0.8, "coverage": 0.9,
                    "macro_f1": 0.75, "kappa": 0.65}})
        self.write_lines("predictions.jsonl", [
            {"t": 0.0, "target": "TV", "score": 0.9, "reason": "ok",
             "attention": {"state": "EM_FOCO_TV", "ear": 0.3, "gaze_activity": 0.02},
             "cameras": {role: {"target": "TV", "score": 0.8, "reason": "ok",
                                "tracking_valid": True, "ear": 0.3}
                         for role in ("notebook", "nao", "validation")},
             "screen_gaze": {"tv:notebook": [100, 200]}},
            {"t": 0.2, "target": "NAO", "score": 0.8, "reason": "ok",
             "cameras": {"notebook": {"target": "INDETERMINADO", "score": 0}},
             "screen_gaze": {"robot:nao": [0.5, 0.5, "head"]}},
        ])
        self.write_lines("frames.jsonl", [
            {"role": "validation", "t": 0.1, "file": "validation/1.jpg",
             "sequence": 1, "received_t": 0.12, "sync_uncertainty_s": 0.01,
             "device_timestamp": None}
        ])
        self.write_lines("events.jsonl", [
            {"t": 0.5, "kind": "INICIO_CENA", "note": "cena 1"}
        ])
        self.write_lines("attention_events.jsonl", [
            {"start": 0, "end": 1.5, "duration": 1.5, "state": "EM_FOCO_TV",
             "duration_class": "CURTO", "supporting_cameras": ["notebook", "validation"],
             "data_loss_percent": {"notebook": 0, "nao": 20, "validation": 0}}
        ])

    def tearDown(self):
        self.temporary.cleanup()

    def write_json(self, name, value):
        (self.session / name).write_text(json.dumps(value), encoding="utf-8")

    def write_lines(self, name, rows):
        (self.session / name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_sync_normalizes_raw_files_and_builds_dashboard(self):
        store = AnalysisStore(self.root)
        self.assertEqual(store.sync_all(), 1)
        self.assertEqual(store.sync_all(), 0)
        result = store.dashboard(self.session_id)
        self.assertEqual(result["overview"]["sessions"], 1)
        self.assertEqual(result["overview"]["participants"], 1)
        self.assertEqual(result["selected"]["frames"], 1)
        self.assertEqual(result["selected"]["cameras"]["validation"]["samples"], 1)
        self.assertEqual(result["selected"]["cameras"]["validation"]["data_loss"], 0)
        self.assertEqual(result["selected"]["attention_states"][0]["state"], "EM_FOCO_TV")
        self.assertEqual(result["selected"]["events"][0]["kind"], "INICIO_CENA")
        self.assertEqual(result["selected"]["reports"][0]["accuracy"], 0.75)
        self.assertEqual(len(result["selected"]["reports"]), 2)
        self.assertAlmostEqual(sum(result["selected"]["target_seconds"].values()), 2.0)

    def test_resync_replaces_rows_instead_of_duplicating_them(self):
        store = AnalysisStore(self.root)
        store.sync_session(self.session_id)
        store.sync_session(self.session_id)
        result = store.dashboard(self.session_id)
        self.assertEqual(result["selected"]["predictions"], 2)
        self.assertEqual(result["selected"]["frames"], 1)


if __name__ == "__main__":
    unittest.main()
