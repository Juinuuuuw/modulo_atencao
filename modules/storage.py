"""Indice SQLite reconstruivel para consulta e analise das sessoes.

Os JSON/JSONL e JPEG continuam sendo o registro bruto do experimento. Este
modulo os normaliza em SQLite para que o painel possa consultar muitas sessoes
sem carregar todos os arquivos na memoria.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


TARGETS = ("NOTEBOOK", "TV", "NAO", "OUTRO_LOCAL", "INDETERMINADO")
ROLES = ("notebook", "nao", "validation")


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def _read_lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class AnalysisStore:
    """Catalogo analitico local; abre uma conexao curta por operacao."""

    def __init__(self, sessions_dir, database_path=None):
        self.sessions_dir = Path(sessions_dir)
        self.database_path = Path(database_path or self.sessions_dir / "observa.sqlite3")
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self):
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 15000")
        return connection

    @contextmanager
    def transaction(self):
        connection = self.connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self):
        with self.transaction() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    participant TEXT NOT NULL,
                    started_utc TEXT NOT NULL,
                    duration REAL,
                    status TEXT NOT NULL,
                    pilot_override TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL,
                    synced_utc TEXT NOT NULL,
                    source_mtime_ns INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    sample_index INTEGER NOT NULL,
                    t REAL NOT NULL,
                    target TEXT NOT NULL,
                    score REAL,
                    reason TEXT,
                    regions_json TEXT,
                    screen_gaze_json TEXT,
                    attention_state TEXT,
                    ear REAL,
                    gaze_activity REAL,
                    PRIMARY KEY (session_id, sample_index)
                );
                CREATE INDEX IF NOT EXISTS predictions_session_time
                    ON predictions(session_id, t);
                CREATE INDEX IF NOT EXISTS predictions_target
                    ON predictions(target);
                CREATE TABLE IF NOT EXISTS camera_predictions (
                    session_id TEXT NOT NULL,
                    sample_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    target TEXT,
                    score REAL,
                    reason TEXT,
                    tracking_valid INTEGER NOT NULL DEFAULT 0,
                    ear REAL,
                    details_json TEXT NOT NULL,
                    PRIMARY KEY (session_id, sample_index, role),
                    FOREIGN KEY (session_id, sample_index)
                        REFERENCES predictions(session_id, sample_index) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS camera_predictions_role
                    ON camera_predictions(session_id, role);
                CREATE TABLE IF NOT EXISTS frames (
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    t REAL NOT NULL,
                    file TEXT NOT NULL,
                    received_t REAL,
                    sync_uncertainty_s REAL,
                    device_timestamp REAL,
                    PRIMARY KEY (session_id, role, sequence)
                );
                CREATE INDEX IF NOT EXISTS frames_session_time ON frames(session_id, t);
                CREATE TABLE IF NOT EXISTS events (
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    event_index INTEGER NOT NULL,
                    t REAL NOT NULL,
                    kind TEXT NOT NULL,
                    note TEXT NOT NULL,
                    PRIMARY KEY (session_id, event_index)
                );
                CREATE TABLE IF NOT EXISTS attention_events (
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    event_index INTEGER NOT NULL,
                    start REAL NOT NULL,
                    end REAL NOT NULL,
                    duration REAL NOT NULL,
                    state TEXT NOT NULL,
                    duration_class TEXT NOT NULL,
                    supporting_cameras_json TEXT NOT NULL,
                    data_loss_json TEXT NOT NULL,
                    PRIMARY KEY (session_id, event_index)
                );
                CREATE INDEX IF NOT EXISTS attention_events_state
                    ON attention_events(session_id, state);
                CREATE TABLE IF NOT EXISTS annotations (
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    annotator TEXT NOT NULL,
                    interval_index INTEGER NOT NULL,
                    start REAL NOT NULL,
                    end REAL NOT NULL,
                    target TEXT NOT NULL,
                    PRIMARY KEY (session_id, annotator, interval_index)
                );
                CREATE TABLE IF NOT EXISTS reports (
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    annotator TEXT NOT NULL,
                    accuracy REAL,
                    coverage REAL,
                    macro_f1 REAL,
                    kappa REAL,
                    report_json TEXT NOT NULL,
                    PRIMARY KEY (session_id, annotator)
                );
            """)
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(sessions)")}
            if "source_mtime_ns" not in columns:
                connection.execute(
                    "ALTER TABLE sessions ADD COLUMN source_mtime_ns INTEGER NOT NULL DEFAULT 0")
            prediction_columns = {row["name"] for row in connection.execute("PRAGMA table_info(predictions)")}
            for name, kind in (("attention_state", "TEXT"), ("ear", "REAL"), ("gaze_activity", "REAL")):
                if name not in prediction_columns:
                    connection.execute(f"ALTER TABLE predictions ADD COLUMN {name} {kind}")
            camera_columns = {row["name"] for row in connection.execute("PRAGMA table_info(camera_predictions)")}
            if "tracking_valid" not in camera_columns:
                connection.execute("ALTER TABLE camera_predictions ADD COLUMN tracking_valid INTEGER NOT NULL DEFAULT 0")
            if "ear" not in camera_columns:
                connection.execute("ALTER TABLE camera_predictions ADD COLUMN ear REAL")

    @staticmethod
    def _source_mtime(path):
        names = ("session.json", "predictions.jsonl", "frames.jsonl", "events.jsonl", "attention_events.jsonl",
                 "annotations.json", "annotation_metadata.json", "report.json", "reports.json")
        return max((item.stat().st_mtime_ns for name in names
                    if (item := path / name).exists()), default=0)

    def sync_all(self):
        with self.transaction() as connection:
            known = {row["id"]: row["source_mtime_ns"] for row in connection.execute(
                "SELECT id, source_mtime_ns FROM sessions")}
        synced = 0
        for metadata_path in sorted(self.sessions_dir.glob("*/session.json")):
            session_id = metadata_path.parent.name
            if known.get(session_id) != self._source_mtime(metadata_path.parent):
                self.sync_session(session_id)
                synced += 1
        return synced

    def sync_session(self, session_id):
        """Substitui atomicamente o indice de uma sessao pelos arquivos atuais."""
        path = self.sessions_dir / session_id
        metadata = _read_json(path / "session.json")
        if not isinstance(metadata, dict):
            raise ValueError(f"Metadados ausentes para a sessao {session_id}")
        predictions = _read_lines(path / "predictions.jsonl")
        frames = _read_lines(path / "frames.jsonl")
        events = _read_lines(path / "events.jsonl")
        attention_events = _read_lines(path / "attention_events.jsonl")
        annotations = _read_json(path / "annotations.json", {}) or {}
        reports = _read_json(path / "reports.json", {}) or {}
        legacy_report = _read_json(path / "report.json")
        if isinstance(legacy_report, dict) and legacy_report.get("annotator") not in reports:
            reports[str(legacy_report.get("annotator", "ultimo"))] = legacy_report
        synced_utc = datetime.now(timezone.utc).isoformat()
        source_mtime_ns = self._source_mtime(path)

        with self.transaction() as connection:
            connection.execute("""
                INSERT INTO sessions(id, participant, started_utc, duration, status,
                                     pilot_override, metadata_json, synced_utc, source_mtime_ns)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    participant=excluded.participant,
                    started_utc=excluded.started_utc,
                    duration=excluded.duration,
                    status=excluded.status,
                    pilot_override=excluded.pilot_override,
                    metadata_json=excluded.metadata_json,
                    synced_utc=excluded.synced_utc,
                    source_mtime_ns=excluded.source_mtime_ns
            """, (session_id, str(metadata.get("participant", "")),
                  str(metadata.get("started_utc", "")), metadata.get("duration"),
                  str(metadata.get("status", "unknown")), str(metadata.get("pilot_override", "")),
                  _json(metadata), synced_utc, source_mtime_ns))
            for table in ("camera_predictions", "predictions", "frames", "events", "attention_events", "annotations", "reports"):
                connection.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))

            for index, row in enumerate(predictions):
                connection.execute("""
                    INSERT INTO predictions(session_id, sample_index, t, target, score, reason,
                                            regions_json, screen_gaze_json, attention_state, ear, gaze_activity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, index, float(row["t"]), str(row.get("target", "INDETERMINADO")),
                      row.get("score"), row.get("reason"), _json(row.get("regions")),
                      _json(row.get("screen_gaze")), row.get("attention", {}).get("state"),
                      row.get("attention", {}).get("ear"), row.get("attention", {}).get("gaze_activity")))
                for role, camera in (row.get("cameras") or {}).items():
                    if isinstance(camera, dict):
                        connection.execute("""
                            INSERT INTO camera_predictions(session_id, sample_index, role, target,
                                                           score, reason, tracking_valid, ear, details_json)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (session_id, index, role, camera.get("target"), camera.get("score"),
                              camera.get("reason"), int(bool(camera.get("tracking_valid"))),
                              camera.get("ear"), _json(camera)))

            connection.executemany("""
                INSERT INTO frames(session_id, role, sequence, t, file, received_t,
                                   sync_uncertainty_s, device_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [(session_id, row["role"], int(row["sequence"]), float(row["t"]), row["file"],
                    row.get("received_t"), row.get("sync_uncertainty_s"), row.get("device_timestamp"))
                   for row in frames])
            connection.executemany("""
                INSERT INTO events(session_id, event_index, t, kind, note) VALUES (?, ?, ?, ?, ?)
            """, [(session_id, index, float(row["t"]), str(row["kind"]), str(row.get("note", "")))
                   for index, row in enumerate(events)])
            connection.executemany("""
                INSERT INTO attention_events(session_id, event_index, start, end, duration, state,
                                             duration_class, supporting_cameras_json, data_loss_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [(session_id, index, float(row["start"]), float(row["end"]), float(row["duration"]),
                    str(row["state"]), str(row["duration_class"]), _json(row.get("supporting_cameras", [])),
                    _json(row.get("data_loss_percent", {}))) for index, row in enumerate(attention_events)])
            for annotator, intervals in annotations.items():
                connection.executemany("""
                    INSERT INTO annotations(session_id, annotator, interval_index, start, end, target)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, [(session_id, annotator, index, float(row["start"]), float(row["end"]), row["target"])
                       for index, row in enumerate(intervals)])
            for annotator, report in reports.items():
                if not isinstance(report, dict):
                    continue
                connection.execute("""
                    INSERT INTO reports(session_id, annotator, accuracy, coverage, macro_f1, kappa, report_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (session_id, annotator, report.get("accuracy"), report.get("coverage"),
                      report.get("macro_f1"), report.get("kappa"), _json(report)))

    @staticmethod
    def _durations(rows, duration):
        totals = {target: 0.0 for target in TARGETS}
        duration = max(0.0, float(duration or 0))
        cursor = 0.0
        for index, row in enumerate(rows):
            start = max(0.0, min(duration, float(row["t"])))
            if start > cursor:
                totals["INDETERMINADO"] += start - cursor
            next_t = float(rows[index + 1]["t"]) if index + 1 < len(rows) else duration
            end = min(duration, start + 0.5, next_t)
            if end > start:
                target = row["target"] if row["target"] in totals else "INDETERMINADO"
                totals[target] += end - start
                cursor = max(cursor, end)
        if cursor < duration:
            totals["INDETERMINADO"] += duration - cursor
        return {key: round(value, 6) for key, value in totals.items()}

    def dashboard(self, session_id=None):
        with self.transaction() as connection:
            session_rows = connection.execute("""
                SELECT s.id, s.participant, s.started_utc, s.duration, s.status,
                       (SELECT COUNT(*) FROM predictions p WHERE p.session_id = s.id) AS predictions,
                       (SELECT COUNT(*) FROM frames f WHERE f.session_id = s.id) AS frames
                FROM sessions s
                ORDER BY s.started_utc DESC
            """).fetchall()
            sessions = [dict(row) for row in session_rows]
            complete = [row for row in sessions if row["status"] != "recording"]
            if session_id is None and complete:
                session_id = complete[0]["id"]
            selected = next((row for row in sessions if row["id"] == session_id), None)
            participants = len({row["participant"] for row in sessions})
            result = {"database": self.database_path.name, "sessions": sessions,
                      "overview": {"sessions": len(sessions), "participants": participants,
                                   "duration": sum(float(row["duration"] or 0) for row in sessions),
                                   "predictions": sum(int(row["predictions"] or 0) for row in sessions)},
                      "selected": None}
            if selected is None:
                return result

            rows = connection.execute("""
                SELECT sample_index, t, target, score FROM predictions
                WHERE session_id = ? ORDER BY sample_index
            """, (session_id,)).fetchall()
            prediction_rows = [dict(row) for row in rows]
            target_seconds = self._durations(prediction_rows, selected["duration"])
            camera_rows = connection.execute("""
                SELECT role, target, COUNT(*) AS samples, SUM(tracking_valid) AS valid_samples
                FROM camera_predictions WHERE session_id = ? GROUP BY role, target
                ORDER BY role, target
            """, (session_id,)).fetchall()
            cameras = {role: {"samples": 0, "valid_samples": 0, "targets": {target: 0 for target in TARGETS}}
                       for role in ROLES}
            for row in camera_rows:
                role = row["role"]
                if role not in cameras:
                    cameras[role] = {"samples": 0, "valid_samples": 0, "targets": {target: 0 for target in TARGETS}}
                target = row["target"] if row["target"] in TARGETS else "INDETERMINADO"
                cameras[role]["targets"][target] += row["samples"]
                cameras[role]["samples"] += row["samples"]
                cameras[role]["valid_samples"] += row["valid_samples"] or 0
            for value in cameras.values():
                value["data_loss"] = (1 - value["valid_samples"] / value["samples"]
                                      if value["samples"] else None)
            attention_rows = connection.execute("""
                SELECT state, SUM(duration) AS duration, COUNT(*) AS events
                FROM attention_events WHERE session_id = ? GROUP BY state ORDER BY state
            """, (session_id,)).fetchall()
            event_rows = connection.execute("""
                SELECT t, kind, note FROM events WHERE session_id = ? ORDER BY t
            """, (session_id,)).fetchall()
            report_rows = connection.execute("""
                SELECT annotator, accuracy, coverage, macro_f1, kappa, report_json
                FROM reports WHERE session_id = ? ORDER BY annotator
            """, (session_id,)).fetchall()
            reports = []
            for row in report_rows:
                value = dict(row)
                raw = value.pop("report_json", "{}")
                temporal = (json.loads(raw).get("temporal_iou") or {})
                value["temporal_mean_iou"] = temporal.get("mean_best_iou")
                value["temporal_f1_05"] = (temporal.get("thresholds", {}).get("0.5") or {}).get("f1")
                reports.append(value)
            result["selected"] = {**selected, "target_seconds": target_seconds,
                                  "cameras": cameras, "events": [dict(row) for row in event_rows],
                                  "attention_states": [dict(row) for row in attention_rows],
                                  "reports": reports}
            return result

