"""Replay saved validation readings through the live decision (no camera needed).

Run from the repository: python -m tools.replay_screen_gaze --participants P014 P016 P030
Only fit samples build the zones; validation rows are replayed causally, matching
the other cameras within 250 ms and the same fixation. No participant files change.
These correlated calibration checks do not measure accuracy during a live session.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

from modules.multicamera import Experiment
from modules.screen_gaze import ScreenGaze, fit_zone


def replay(root, participants):
    library = ScreenGaze(root / "participants")
    reports = {}
    with tempfile.TemporaryDirectory() as directory:
        exp = Experiment(directory)
        exp.screen_gaze = library
        for participant in participants:
            models = library.models_for(participant)
            if not models:
                raise ValueError(f"No saved maps for {participant}")
            labels = {m["role"]: m["source"] for m in models.values()}
            exp.camera_labels = lambda: labels
            calibrations = []
            for model in models.values():
                cal = json.loads((root / "calibrations" / f"{model['calibration_id']}.json").read_text(encoding="utf-8"))
                model["zone"] = fit_zone(cal["samples"])
                fit_count = len(cal["fit_points"])
                model["cross_zones"] = library._cross_zones({"cross": {
                    role: {**item, "rows": [r for r in item["rows"] if r["step"] < fit_count]}
                    for role, item in cal.get("cross", {}).items()}})
                calibrations.append(cal)
            report = {}
            for cal in calibrations:
                truth = {"notebook": "NOTEBOOK", "tv": "TV", "robot": "NAO"}[cal["surface"]]
                counts = Counter()
                exp.live_targets.clear()
                for row in sorted(cal["checks"], key=lambda r: r["capture"]):
                    now = row["capture"]
                    observations = {cal["role"]: {**row, "uncertainty": .01}}
                    for role, item in cal.get("cross", {}).items():
                        candidates = [r for r in item["rows"] if r["step"] == row["step"] and 0 <= now - r["capture"] <= .25]
                        if candidates:
                            observations[role] = {**max(candidates, key=lambda r: r["capture"]), "uncertainty": .01}
                    exp.observations = observations
                    result = exp.live_gaze(now, participant)
                    surface = result["robot"] if truth == "NAO" else result["screens"][cal["surface"]]
                    own = next(e["gaze"] for e in surface["estimates"] if e["role"] == cal["role"])
                    counts["samples"] += 1
                    counts["target_correct"] += result["target"] == truth
                    counts["own_inside"] += bool(own.get("inside"))
                    counts["own_visible"] += bool(own["visible"])
                    counts["zone_inside_map_outside"] += bool(own.get("in_zone") and not own.get("inside"))
                    counts["wrong_target"] += result["target"] not in (truth, "INDETERMINADO")
                report[cal["surface"]] = dict(counts)
            reports[participant] = report
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=Path, default=Path("sessions"))
    parser.add_argument("--participants", nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(replay(args.sessions, args.participants), indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)
