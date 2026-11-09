"""Estado do experimento e processamento central das câmeras."""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading
import time
import uuid

import cv2
import numpy as np

from .attention_timeline import AttentionConfig, AttentionTimeline
from .target_classifier import TARGETS, UNKNOWN, ROLES, REGIONS, TargetClassifier, feature_status, screen_feature_status, fuse, region_votes
from .validation import evaluate_checks
from .calibration_protocol import PROFILES, calibration_plan
from .screen_gaze import ScreenGaze, SURFACES


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def append_json(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def read_lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass
class Frame:
    image: np.ndarray
    jpeg: bytes
    capture: float
    received: float
    uncertainty: float
    sequence: int
    pose: list | None = None
    device_timestamp: float | None = None


class Experiment:
    def __init__(self, output, extractor_factory=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.nodes = {}
        self.assignments = {role: None for role in ROLES}
        self.frames = {}
        self.frame_times = {}
        self.observations = {}
        self.predictions = {}
        self.classifier = TargetClassifier()
        self.pose_reference = {}
        self.samples = []
        self.checks = []
        self.calibration = None
        self.calibration_participant = None
        self.calibration_protocol = None
        self.displays = {}
        # Mapas de posição por participante, salvos em disco (voltam após reiniciar).
        self.screen_gaze = ScreenGaze(self.output / "participants")
        self.fit_report = None
        self.check_report = None
        self.current = None
        self.live = {"target": UNKNOWN, "score": 0, "reason": "calibre os alvos"}
        self.errors = {}
        self.last_error = None
        self.stop_event = threading.Event()
        self.extractor_factory = extractor_factory
        self.thread = None
        # Último acesso da aba "Olhar ao vivo"; acelera a análise enquanto aberta.
        self.live_viewer_seen = -1e9
        self.last_logged = -1e9
        self.attention_config = AttentionConfig()
        self.attention_tracker = None
        self.attention_live = {"state": "INDETERMINADO", "since": None,
                               "gaze_activity": None, "ear": None}
        # Atribuições lembradas por câmera física (computador/índice). Após reiniciar o
        # servidor, o conector volta a receber as mesmas câmeras sem fechá-las: reabrir
        # webcams no Windows às vezes trava o driver até reiniciar o conector.
        self.remembered_path = self.output / "assignments.json"
        try:
            self.remembered = json.loads(self.remembered_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.remembered = {}

    def start_worker(self):
        self.thread = threading.Thread(target=self._worker, daemon=True, name="face-processing")
        self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=10)
        with self.lock:
            if self.current:
                self.stop_recording()

    def configure(self, mapping):
        if self.current:
            raise ValueError("Encerre a sessão antes de trocar as câmeras")
        if set(mapping) != set(ROLES):
            raise ValueError("Selecione os três papéis")
        selected = [v for v in mapping.values() if v]
        if len(set(selected)) != len(selected):
            raise ValueError("Uma câmera não pode ocupar dois papéis")
        available = {f"{node_id}:{camera['id']}" for node_id, node in self.nodes.items()
                     for camera in node["cameras"]}
        if any(source not in available for source in selected):
            raise ValueError("Câmera não registrada")
        self.assignments = dict(mapping)
        self.remembered = self.camera_labels()
        try:
            write_json(self.remembered_path, self.remembered)
        except OSError:
            pass
        self.classifier = TargetClassifier()
        self.frames.clear()
        self.frame_times.clear()
        self.observations.clear()
        self.predictions.clear()
        self.samples.clear()
        self.checks.clear()
        self.fit_report = self.check_report = self.calibration = None
        self.calibration_participant = None
        self.pose_reference = {}
        self.calibration_protocol = None
        self.screen_gaze.reset()
        self.live = {"target": UNKNOWN, "score": 0, "reason": "configuração alterada; recalibre"}

    def camera_label(self, role):
        """Identificação estável da câmera (computador/índice), que sobrevive à reconexão do cliente."""
        source = self.assignments.get(role)
        if not source:
            return None
        node_id, _, camera_id = source.partition(":")
        node = self.nodes.get(node_id)
        return f"{node['name']}/{camera_id}" if node else source

    def camera_labels(self):
        return {role: self.camera_label(role) for role in ROLES}

    def restore_assignments(self, node_id):
        """Devolve a um conector que se registrou de novo as câmeras que ele tinha."""
        node = self.nodes[node_id]
        restored = []
        for role in ROLES:
            label = self.remembered.get(role)
            if self.assignments.get(role) or not label:
                continue
            for camera in node["cameras"]:
                source = f"{node_id}:{camera['id']}"
                if f"{node['name']}/{camera['id']}" == label and source not in self.assignments.values():
                    self.assignments[role] = source
                    restored.append(role)
        return restored

    def receive(self, source, image, jpeg, capture, uncertainty, sequence, pose=None, device_timestamp=None):
        now = time.perf_counter()
        if not all(math.isfinite(x) for x in (capture, uncertainty)) or not 0 <= uncertainty <= 2:
            raise ValueError("Sincronização inválida")
        if capture > now + 0.1 or now - capture > 2:
            raise ValueError("Quadro atrasado ou relógio dessincronizado; sincronize novamente")
        with self.lock:
            if source not in self.assignments.values():
                return False
            old = self.frames.get(source)
            if old and (sequence <= old.sequence or capture <= old.capture):
                return False
            frame = Frame(image, jpeg, capture, now, uncertainty, sequence, pose, device_timestamp)
            self.frames[source] = frame
            self.frame_times.setdefault(source, deque(maxlen=120)).append(now)
            if self.current:
                role = next(r for r, s in self.assignments.items() if s == source)
                t = capture - self.current["origin"]
                if t >= 0:
                    name = f"{role}/{sequence:012d}.jpg"
                    (self.current["path"] / name).write_bytes(jpeg)
                    append_json(self.current["path"] / "frames.jsonl", {
                        "role": role, "t": t, "file": name, "sequence": sequence,
                        "received_t": now - self.current["origin"],
                        "sync_uncertainty_s": uncertainty, "device_timestamp": device_timestamp})
            return True

    def begin_calibration(self, check=False, participant=None, profile="standard", automatic=False, chain_check=False):
        if self.screen_gaze.active:
            raise ValueError("Conclua ou cancele a calibração de posição na tela do participante")
        if profile not in PROFILES:
            raise ValueError("Perfil de calibração inválido")
        if self.current:
            raise ValueError("Encerre a gravação primeiro")
        if not self.assignments["notebook"] or not self.assignments["nao"]:
            raise ValueError("Selecione as câmeras Notebook e NAO")
        if check and not self.classifier.models:
            raise ValueError("Faça o ajuste dos alvos primeiro")
        if check and participant is not None and participant != self.calibration_participant:
            raise ValueError("O participante mudou. Refaça o ajuste dos alvos.")
        if not check and (not participant or len(participant) > 80):
            raise ValueError("Informe o participante antes de calibrar (até 80 caracteres)")
        if check:
            self.checks = []
            self.check_report = None
        else:
            if self.calibration_participant != participant:
                self.screen_gaze.reset()
            self.calibration_participant = participant
            self.calibration_protocol = {"profile": profile, "automatic": automatic,
                                         "fit_plan": calibration_plan(False, profile), **PROFILES[profile]}
            self.samples = []
            self.checks = []
            self.classifier = TargetClassifier()
            self.fit_report = self.check_report = None
            self.pose_reference = {}
        self.calibration = {"mode": "check" if check else "fit", "plan": calibration_plan(check, profile),
                            "index": 0, "started": None, "error": None, "complete": False,
                            "profile": profile, "automatic": automatic, "paused": False, "revision": 0,
                            "chain_check": chain_check,
                            "run_id": uuid.uuid4().hex, "wait_reason": None, "retries": {}, **PROFILES[profile]}

    def cue(self):
        cal = self.calibration
        if not cal or cal["complete"]:
            return {"active": False, "complete": bool(cal and cal["complete"]),
                    "error": cal["error"] if cal else None}
        return {"active": True, "cue_id": f"{cal['run_id']}:{cal['index']}:{cal['revision']}",
                "step": cal["plan"][cal["index"]], "index": cal["index"], "total": len(cal["plan"]),
                "mode": cal["mode"], "paused": cal["paused"], "collecting": cal["started"] is not None,
                "remaining": max(0, cal["duration"] - (time.perf_counter() - cal["started"])) if cal["started"] is not None else cal["duration"]}

    def report_display(self, data):
        display, client = data["display"], data["client"]
        if display not in ("notebook", "tv") or not isinstance(client, str) or not 1 <= len(client) <= 64:
            raise ValueError("Tela inválida")
        now = time.perf_counter()
        layout = data.get("layout")
        if layout is not None:
            if not isinstance(layout,list) or len(layout)!=4 or any(not isinstance(v,(int,float)) or not math.isfinite(v) or abs(v)>100000 for v in layout) or min(layout[:2])<=0:
                raise ValueError("Dimensões da tela inválidas")
            layout = [int(v) for v in layout]
        self.displays = {key: value for key, value in self.displays.items() if now - value["seen"] < 10}
        if len(self.displays) >= 20 and client not in self.displays:
            raise ValueError("Muitas telas abertas; feche as janelas extras")
        self.displays[client] = {"display": display, "seen": now,
                                "ready": data.get("ready") is True, "visible": data.get("visible") is True,
                                "cue_id": data.get("cue_id"), "fullscreen": data.get("fullscreen") is True,"layout":layout}

    def display_status(self, now=None):
        now = time.perf_counter() if now is None else now
        cue_id = self.cue().get("cue_id")
        return {display: {"online": any(d["display"] == display and d["visible"] and now - d["seen"] < 1.0
                                        for d in self.displays.values()),
                          "ready": any(d["display"] == display and d["visible"] and d["ready"]
                                       and d["cue_id"] == cue_id and now - d["seen"] < 1.0
                                       for d in self.displays.values())} for display in ("notebook", "tv")}

    def _discard_step(self):
        cal = self.calibration
        rows = self.samples if cal["mode"] == "fit" else self.checks
        rows[:] = [row for row in rows if row["step"] != cal["index"]]
        cal["started"] = None
        cal["revision"] += 1

    def pause_calibration(self, paused):
        if not self.calibration or self.calibration["complete"]:
            raise ValueError("Não há calibração em andamento")
        self._discard_step()
        self.calibration.update(paused=paused, error=None)

    def cancel_calibration(self):
        if self.current:
            raise ValueError("Encerre a gravação antes de trocar o participante")
        self.calibration = self.calibration_protocol = self.calibration_participant = None
        self.fit_report = self.check_report = None
        self.samples, self.checks = [], []
        self.classifier = TargetClassifier()
        self.pose_reference = {}
        self.screen_gaze.reset()

    def capture_step(self, index=None):
        if self.screen_gaze.active:
            raise ValueError("Aguarde ou cancele a calibração de posição")
        cal = self.calibration
        if not cal or cal["started"] is not None:
            raise ValueError("Inicie a calibração ou aguarde a coleta atual")
        if cal["automatic"] and not cal["paused"] and not cal["complete"]:
            raise ValueError("Pause a sequência automática antes de coletar manualmente")
        cal["automatic"] = False
        cal["paused"] = False
        if index is not None:
            if not 0 <= index < len(cal["plan"]):
                raise ValueError("Etapa inválida")
            cal["index"] = index
        if cal["index"] >= len(cal["plan"]):
            raise ValueError("Etapas concluídas")
        rows = self.checks if cal["mode"] == "check" else self.samples
        rows[:] = [r for r in rows if r["step"] != cal["index"]]
        cal.update(started=time.perf_counter(), error=None, complete=False, revision=cal["revision"] + 1)
        if cal["mode"] == "fit":
            self.classifier = TargetClassifier()
            self.fit_report = self.check_report = None
            self.checks = []
            self.pose_reference = {}
        else:
            self.check_report = None

    def _advance_calibration(self, now):
        cal = self.calibration
        if not cal or cal["complete"] or cal["paused"]:
            return
        if cal["automatic"]:
            missing = [d for d, value in self.display_status(now).items() if not value["ready"]]
            missing_cameras = [role for role in ROLES if self.assignments[role]
                               if not self.frames.get(self.assignments[role])
                               or now - self.frames[self.assignments[role]].capture > 0.7]
            if missing or missing_cameras:
                if cal["started"] is not None:
                    self._discard_step()
                cal["wait_reason"] = ("Aguardando tela visível/instrução: " + ", ".join(missing) if missing
                                      else "Aguardando imagem: " + ", ".join(missing_cameras))
                return
            cal["wait_reason"] = None
            if cal["started"] is None:
                cal["started"] = now
                return
        if cal["started"] is None or now - cal["started"] < cal["duration"]:
            return
        if (cal["automatic"] or cal["profile"] == "assisted") and cal["mode"] == "fit":
            counts = [sum(s["step"] == cal["index"] and s["role"] == role for s in self.samples)
                      for role in ROLES]
            if max(counts) < 4:
                attempts = cal["retries"].get(cal["index"], 0) + 1
                cal["retries"][cal["index"]] = attempts
                self._discard_step()
                if not cal["automatic"]:
                    cal["error"] = "Poucas leituras nesta etapa. Ajuste o rosto/enquadramento e clique em coletar novamente."
                    return
                if attempts >= 2:
                    cal.update(paused=True, error="Poucas leituras válidas nesta etapa. Ajuste o enquadramento e retome.")
                return
        cal["started"] = None
        cal["index"] += 1
        if cal["index"] < len(cal["plan"]):
            return
        self.finish_calibration()
        if (cal["automatic"] or cal["chain_check"]) and cal["mode"] == "fit" and self.fit_report:
            self.begin_calibration(check=True, participant=self.calibration_participant,
                                   profile=cal["profile"], automatic=cal["automatic"], chain_check=cal["chain_check"])

    def finish_calibration(self):
        cal = self.calibration
        if not cal or cal["started"] is not None:
            raise ValueError("Aguarde a coleta terminar")
        cal["complete"] = True
        cal["index"] = len(cal["plan"])
        cal["error"] = None
        if cal["mode"] == "fit":
            try:
                missing_steps = [i + 1 for i in range(len(cal["plan"]))
                                 if max(sum(s["step"] == i and s["role"] == role for s in self.samples)
                                        for role in ROLES) < 4]
                if missing_steps:
                    raise ValueError("Refaça as etapas com poucas leituras: " + ", ".join(map(str, missing_steps)))
                self.fit_report = self.classifier.fit(self.samples)
                for role in self.classifier.models:
                    poses = [s["pose"] for s in self.samples if s["role"] == role and s.get("pose")]
                    if poses:
                        spread = np.ptp(poses, axis=0)
                        if float(max(spread)) > 0.10:
                            raise ValueError("Cabeça do NAO moveu durante a calibração. Mantenha-a fixa e repita.")
                        self.pose_reference[role] = np.median(poses, axis=0).tolist()
            except ValueError as error:
                self.classifier = TargetClassifier()
                self.fit_report = None
                cal["error"] = str(error)
        else:
            self.check_report = evaluate_checks(self.checks)
            counts = {target: sum(r["expected"] == target for r in self.checks) for target in TARGETS}
            self.check_report["counts"] = counts
            self.check_report["step_counts"] = [sum(r["step"] == i for r in self.checks) for i in range(len(cal["plan"]))]
            self.check_report["sufficient"] = all(n >= 5 for n in counts.values()) and all(n >= 5 for n in self.check_report["step_counts"])
            if self.calibration_protocol:
                self.calibration_protocol["check_plan"] = cal["plan"]

    def _worker(self):
        if self.extractor_factory is None:
            from .visual_attention_monitor import FaceLandmarkExtractor
            factory = FaceLandmarkExtractor
        else:
            factory = self.extractor_factory
        extractors, processed, last_run = {}, {}, {}
        try:
            while not self.stop_event.is_set():
                tick = time.perf_counter()
                with self.lock:
                    # Na calibração de posição só uma câmera importa: ela roda a 30 Hz
                    # e as demais a ~5 Hz. Analisar as três a 30 Hz travava o servidor.
                    focus = self.screen_gaze.calibration["role"] if self.screen_gaze.active else None
                for role in ROLES:
                    if focus and role != focus and tick - last_run.get(role, -1e9) < 0.2:
                        continue
                    with self.lock:
                        source = self.assignments[role]
                        frame = self.frames.get(source)
                    if not frame or processed.get(role) == (source, frame.sequence):
                        continue
                    processed[role] = (source, frame.sequence)
                    last_run[role] = tick
                    try:
                        if role not in extractors:
                            extractors[role] = factory()
                        observation = extractors[role].extract(frame.image, frame.capture)
                        vector, quality_reason = feature_status(observation)
                        screen_features, screen_quality_reason = screen_feature_status(observation)
                        with self.lock:
                            if self.assignments[role] != source:
                                continue
                            self.errors.pop(role, None)
                            self.observations[role] = {"features": vector, "capture": frame.capture,
                                                       "screen_features": screen_features,
                                                       "screen_quality_reason": screen_quality_reason,
                                                       "uncertainty": frame.uncertainty, "pose": frame.pose,
                                                       "sequence": frame.sequence,
                                                       "eyes": getattr(observation, "eye_centers", None),
                                                       "quality_reason": quality_reason,
                                                       "face_count": getattr(observation, "face_count", 1),
                                                       "glabella": getattr(observation,"glabella",None),
                                                       "iris_diameter": getattr(observation,"iris_diameter",None),
                                                       "ear": getattr(observation,"ear",None),
                                                       "direction": getattr(observation, "gaze_direction", None)}
                            cal = self.calibration
                            if (cal and cal["mode"] == "fit" and cal["started"] is not None
                                    and cal["started"] + cal["settling"] <= frame.capture <= cal["started"] + cal["duration"]
                                    and vector is not None and frame.uncertainty <= 0.15):
                                self.samples.append({"role": role, "target": cal["plan"][cal["index"]]["target"],
                                                     "features": vector, "step": cal["index"], "pose": frame.pose,
                                                     "ear": getattr(observation, "ear", None),
                                                     "capture": frame.capture, "sequence": frame.sequence,
                                                     "sync_uncertainty_s": frame.uncertainty,
                                                     "part": cal["plan"][cal["index"]].get("part")})
                    except Exception as error:
                        with self.lock:
                            self.errors[role] = str(error)
                            self.observations.pop(role, None)
                try:
                    with self.lock:
                        self.tick(time.perf_counter())
                except OSError as error:
                    with self.lock:
                        self.last_error = f"Falha ao gravar no disco: {error}. A sessão foi interrompida."
                        if self.current:
                            self.current["metadata"].update(status="interrupted", duration=time.perf_counter() - self.current["origin"])
                            try:
                                write_json(self.current["path"] / "session.json", self.current["metadata"])
                            except OSError:
                                pass
                            self.current = None
                # Durante calibrações, processar todos os quadros (~30 Hz), como o módulo
                # isolado: mais amostras por ponto e rastreamento contínuo do FaceMesh.
                with self.lock:
                    calibrating = self.screen_gaze.active or bool(
                        self.calibration and self.calibration.get("started") is not None)
                    watching = time.perf_counter() - self.live_viewer_seen < 3
                # Com a aba "Olhar ao vivo" aberta, ~15 Hz para o ponto acompanhar o olhar.
                period = 1 / 30 if calibrating else 1 / 15 if watching else 0.20
                self.stop_event.wait(max(0.005, period - (time.perf_counter() - tick)))
        finally:
            for extractor in extractors.values():
                extractor.close()

    def tick(self, now):
        self.screen_gaze.tick(now, self.observations, self.camera_labels(), self.displays)
        cal_gaze=self.screen_gaze.calibration
        if cal_gaze and cal_gaze["phase"] in ("complete","failed"):
            signature=(cal_gaze["id"],cal_gaze["phase"])
            if getattr(self,"last_gaze_diagnostic",None)!=signature:
                directory=self.output/"calibrations"
                directory.mkdir(exist_ok=True)
                write_json(directory/f"{cal_gaze['id']}.json",self.screen_gaze.diagnostic())
                self.last_gaze_diagnostic=signature
        predictions = {}
        captures = [o["capture"] for o in self.observations.values() if now - o["capture"] <= 0.7]
        for role in ROLES:
            # A terceira câmera (tela de apresentação) é opcional na fusão.
            if not self.assignments.get(role) and role not in self.observations:
                continue
            observation = self.observations.get(role)
            reason = None
            if not observation or now - observation["capture"] > 0.7:
                reason = "câmera sem leitura recente"
            elif observation["uncertainty"] > 0.15:
                reason = "incerteza de sincronização alta"
            elif captures and max(captures) - observation["capture"] > 0.25:
                reason = "quadros fora da janela de fusão"
            elif role in self.pose_reference:
                pose = observation.get("pose")
                if pose is None or max(abs(a - b) for a, b in zip(pose, self.pose_reference[role])) > 0.08:
                    reason = "cabeça do NAO mudou de pose; recalibre"
            tracking_valid = reason is None and observation is not None and observation.get("features") is not None
            prediction = ({"target": UNKNOWN, "score": 0, "reason": reason} if reason
                          else self.classifier.predict(role, observation["features"]))
            prediction.update(tracking_valid=tracking_valid,
                              ear=observation.get("ear") if observation else None,
                              face_count=observation.get("face_count") if observation else 0,
                              sync_uncertainty_s=observation.get("uncertainty") if observation else None)
            predictions[role] = prediction
        self.predictions = predictions
        self.live = fuse(list(predictions.values()))
        cal = self.calibration
        if (cal and cal["mode"] == "check" and cal["started"] is not None
                and cal["started"] + cal["settling"] <= now <= cal["started"] + cal["duration"]):
            self.checks.append({"step": cal["index"], "expected": cal["plan"][cal["index"]]["target"],
                                "elapsed": now - cal["started"], **self.live})
        self._advance_calibration(now)
        # O registro mantém ~5 Hz mesmo quando a análise roda mais rápido.
        if self.current and now - self.last_logged >= 0.19:
            self.last_logged = now
            screens = {}
            labels = self.camera_labels()
            for key in self.screen_gaze.models:
                position = self.screen_gaze.position(key, now, self.observations, labels, self.displays, require_view=False)
                screens[key] = ([position["x"], position["y"]] + ([position["part"]] if "part" in position else [])
                                if position["visible"] else None)
            t = now - self.current["origin"]
            tracking = {role: bool(value.get("tracking_valid")) for role, value in predictions.items()}
            ears = [value["ear"] for value in predictions.values()
                    if value.get("tracking_valid") and value.get("ear") is not None]
            ear = float(np.median(ears)) if ears else None
            supporting = [role for role, value in predictions.items()
                          if value.get("tracking_valid") and value.get("target") == self.live["target"]]
            gaze_point = self._attention_gaze_point(self.live["target"], screens)
            closed, attention = self.attention_tracker.update(
                t, self.live["target"], ear=ear, tracking=tracking,
                supporting_cameras=supporting, gaze_point=gaze_point)
            self.attention_live = attention
            for event in closed:
                append_json(self.current["path"] / "attention_events.jsonl", event)
            append_json(self.current["path"] / "predictions.jsonl", {
                "t": t, **self.live, "cameras": predictions, "screen_gaze": screens,
                "attention": attention})

    @staticmethod
    def _attention_gaze_point(target, screens):
        surface = {"NOTEBOOK": "notebook", "TV": "tv", "NAO": "robot"}.get(target)
        if surface is None:
            return None
        points = [value[:2] for key, value in screens.items()
                  if key.startswith(surface + ":") and value is not None]
        return np.median(np.asarray(points, dtype=float), axis=0).tolist() if points else None

    def live_gaze(self, now=None, participant=None):
        """Para onde a pessoa olha, pelas calibrações de posição do participante escolhido.

        Cada câmera calibrada vota na superfície (notebook, TV, robô) onde a sua
        estimativa cai; sem estimativa dentro de nenhuma, vota em "outro local".
        Sem calibrações de posição, usa o classificador de alvos, se houver.
        """
        now = time.perf_counter() if now is None else now
        self.live_viewer_seen = now
        gaze = self.screen_gaze
        participant = participant or gaze.participant or self.calibration_participant
        labels = self.camera_labels()
        surfaces = {surface: {"calibrated": False, "estimates": []} for surface in SURFACES}
        # known[câmera][superfície] = True (estimativa dentro) / False (leitura válida, fora).
        known, parts = {}, {}
        for key in sorted(gaze.models_for(participant)):
            surface, role = key.split(":")
            position = gaze.position(key, now, self.observations, labels, self.displays,
                                     require_view=False, participant=participant)
            surfaces[surface]["calibrated"] = True
            surfaces[surface]["estimates"].append({**gaze.summary(key, participant), "gaze": position})
            if position["visible"] or position.get("outside"):
                inside = bool(position.get("inside"))
                known.setdefault(role, {})[surface] = inside
                if inside and surface == "robot":
                    parts[role] = position["part"]
        names = {"notebook": "NOTEBOOK", "tv": "TV", "robot": "NAO"}
        # Cada câmera só opina sobre as superfícies que calibrou: a favor quando a
        # estimativa cai só nela; contra quando a leitura é válida e cai fora dela.
        # Uma câmera sem mapa do robô não contradiz a câmera que vê o olhar no robô.
        balance = {}
        for surface in SURFACES:
            support = sum(1 for seen in known.values() if seen.get(surface) is True and sum(seen.values()) == 1)
            against = sum(1 for seen in known.values() if seen.get(surface) is False)
            if support:
                balance[surface] = (support - against, support, against)
        if any(item["calibrated"] for item in surfaces.values()):
            best = max((b[0] for b in balance.values()), default=None)
            winners = [s for s, b in balance.items() if b[0] == best]
            if best is not None and best > 0 and len(winners) == 1:
                surface = winners[0]
                target = names[surface]
                _, support, against = balance[surface]
                reason = f"{support} câmera(s) a favor" + (f", {against} contra" if against else "")
                score = support / len(known)
            elif balance:
                target, reason, score = UNKNOWN, "câmeras discordam", 0
            elif known:
                target, reason, score = "OUTRO_LOCAL", "fora das telas e do robô calibrados", 1
            else:
                target, reason, score = UNKNOWN, "sem leitura válida das câmeras calibradas", 0
            seen_parts = [part for part in parts.values() if part]
            part = max(set(seen_parts), key=seen_parts.count) if target == "NAO" and seen_parts else None
            source = "posição"
        else:
            target, reason, score = self.live["target"], self.live["reason"], self.live["score"]
            part = self.live.get("part") if target == "NAO" else None
            source = "classificador" if self.classifier.models else "nenhuma"
        surfaces["robot"].update(looking=target == "NAO", part=part)
        return {"target": target, "score": round(float(score), 3), "reason": reason, "source": source,
                "participant": participant, "participants": gaze.participants(),
                "classifier_ready": bool(self.classifier.models),
                "screens": {d: surfaces[d] for d in ("notebook", "tv")}, "robot": surfaces["robot"],
                "recording": bool(self.current),
                "cameras": {role: bool(self.frames.get(source) and now - self.frames[source].capture < 0.7)
                            for role, source in self.assignments.items() if source}}

    def eye_overlay(self, role, frame):
        """Metadados leves para desenhar no navegador, preservando os JPEG originais."""
        observation = self.observations.get(role)
        camera = self.predictions.get(role, {})
        fresh = bool(observation and time.perf_counter() - observation["capture"] <= 0.35
                     and abs(frame.capture - observation["capture"]) <= 0.30)
        geometry = observation if fresh else {}
        valid = fresh and observation["features"] is not None
        calibrated = role in self.classifier.models
        target = self.live["target"]
        reason = self.live["reason"]
        if not fresh:
            reason = "Aguardando leitura dos olhos"
        elif not valid:
            reason = observation.get("quality_reason") or "Leitura dos olhos insuficiente"
        elif not calibrated:
            reason = "Calibre esta câmera"
        elif camera.get("target", UNKNOWN) == UNKNOWN:
            reason = "Leitura desta câmera incerta"
        visible_eyes = fresh and bool(geometry.get("eyes"))
        calibrated_fusion = bool(self.classifier.models)
        certain = bool(visible_eyes and calibrated_fusion and target != UNKNOWN)
        regions = {}
        for key in REGIONS:
            group = region_votes(self.live)[key]
            # A cor é o resultado da fusão para a mesma pessoa; uma câmera
            # auxiliar não precisa, sozinha, distinguir todos os alvos.
            group_certain = visible_eyes and calibrated_fusion and group["target"] != UNKNOWN
            regions[key] = {**group, "target": group["target"] if group_certain else UNKNOWN,
                            "reason": group["reason"] if group_certain else reason}
        return {"eyes": geometry.get("eyes"), "direction": geometry.get("direction") if valid else None,
                "target": target if certain else UNKNOWN, "reason": reason,
                "regions": regions,
                "color_basis": "fused",
                "calibrated": calibrated, "fresh": fresh,
                "observation_sequence": geometry.get("sequence"), "kind": "illustrative_2d"}

    def start_recording(self, participant, override=""):
        if self.screen_gaze.active:
            raise ValueError("Conclua ou cancele a calibração de posição antes de gravar")
        if self.current:
            raise ValueError("Já existe uma sessão em gravação")
        if not participant or len(participant) > 80:
            raise ValueError("Informe um identificador de participante com até 80 caracteres")
        if participant != self.calibration_participant:
            raise ValueError("Calibre os alvos para este participante antes de gravar")
        if not self.classifier.models or not self.check_report or not self.check_report["sufficient"]:
            raise ValueError("Conclua a calibração e a verificação de todos os alvos")
        if self.calibration and self.calibration["started"] is not None:
            raise ValueError("Aguarde a etapa atual terminar")
        for role, source in self.assignments.items():
            frame = self.frames.get(source)
            if not frame or time.perf_counter() - frame.capture > 0.7:
                raise ValueError(f"Câmera {role} sem sinal recente")
        report = self.check_report
        if (report["accuracy"] < 0.65 or report["coverage"] < 0.65) and not override.strip():
            raise ValueError("Verificação abaixo de 65%. Recalibre ou registre uma justificativa de uso em piloto.")
        session_id = uuid.uuid4().hex
        path = self.output / session_id
        path.mkdir()
        for role in ROLES:
            (path / role).mkdir()
        metadata = {"id": session_id, "participant": participant,
                    "started_utc": datetime.now(timezone.utc).isoformat(), "duration": None,
                    "status": "recording", "assignments": dict(self.assignments),
                    "calibration_participant": self.calibration_participant,
                    "calibration_protocol": self.calibration_protocol,
                    "screen_gaze_models": self.screen_gaze.export(),
                    "nodes": {k: {"name": v["name"], "cameras": v["cameras"]} for k, v in self.nodes.items()},
                    "calibration": self.samples, "models": self.classifier.models,
                    "pose_reference": self.pose_reference, "fit_report": self.fit_report,
                    "check_samples": self.checks, "check_report": self.check_report,
                    "attention_method": {
                        "config": self.attention_config.export(),
                        "low_activity_label": "indicador observável; não infere distração cognitiva",
                        "fusion": "todas as câmeras válidas; não há câmera ativa única"},
                    "pilot_override": override.strip(),
                    "timing": "captura aproximada após leitura, corrigida por RTT; não é sincronização de hardware"}
        write_json(path / "session.json", metadata)
        write_json(path / "annotations.json", {})
        self.current = {"id": session_id, "path": path, "origin": time.perf_counter(), "metadata": metadata}
        ears = [float(row["ear"]) for row in self.samples
                if row.get("ear") is not None and math.isfinite(float(row["ear"]))]
        reference = float(np.median(ears)) if ears else None
        self.attention_tracker = AttentionTimeline(self.attention_config, reference, ROLES)
        metadata["attention_method"].update(ear_reference=reference,
                                            ear_threshold=self.attention_tracker.ear_threshold)
        write_json(path / "session.json", metadata)
        self.attention_live = {"state": "INDETERMINADO", "since": 0,
                               "gaze_activity": None, "ear": None,
                               "ear_threshold": self.attention_tracker.ear_threshold}
        self.last_logged = -1e9
        self.calibration = None
        return session_id

    def event(self, kind, note):
        if not self.current:
            raise ValueError("Inicie uma gravação")
        if not kind or len(kind) > 80 or len(note) > 500:
            raise ValueError("Evento inválido")
        append_json(self.current["path"] / "events.jsonl", {
            "t": time.perf_counter() - self.current["origin"], "kind": kind, "note": note})

    def stop_recording(self):
        if not self.current:
            raise ValueError("Nenhuma gravação ativa")
        current = self.current
        duration = time.perf_counter() - current["origin"]
        if self.attention_tracker:
            for event in self.attention_tracker.close(duration):
                append_json(current["path"] / "attention_events.jsonl", event)
        current["metadata"].update(duration=duration, status="complete")
        write_json(current["path"] / "session.json", current["metadata"])
        self.current = None
        self.attention_tracker = None
        return current["id"]

    def reading_status(self, role, now=None, spatial=False):
        now = time.perf_counter() if now is None else now
        obs = self.observations.get(role)
        if not obs or now - obs["capture"] > .7:
            return {"valid": False, "reason": "Sem análise recente desta câmera"}
        reason = ("Incerteza de sincronização alta" if obs.get("uncertainty", 1) > .15
                  else obs.get("screen_quality_reason",obs.get("quality_reason")) if spatial else obs.get("quality_reason"))
        vector = obs.get("screen_features",obs.get("features")) if spatial else obs.get("features")
        valid = vector is not None and reason is None
        return {"valid": valid, "reason": reason or ("Olhos detectados; leitura válida" if valid else "Sem leitura válida dos olhos"),
                "face_count": obs.get("face_count"), "age": now-obs["capture"]}

    def state(self):
        now = time.perf_counter()
        def camera_stats(source):
            times = self.frame_times.get(source, ())
            recent = [t for t in times if now - t <= 2]
            fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 and recent[-1] > recent[0] else 0
            frame = self.frames.get(source)
            online = bool(frame and now - frame.capture < 0.7)
            return {"fps": round(fps, 1) if online else 0,
                    "transport_ms": round(max(0, frame.received - frame.capture) * 1000, 1) if frame else None}
        cal = self.calibration
        calibration = None
        if cal:
            calibration = {**cal, "remaining": max(0, cal["duration"] - (now - cal["started"])) if cal["started"] else 0,
                           "counts": {role: {target: sum(s["role"] == role and s["target"] == target for s in self.samples)
                                             for target in TARGETS} for role in ROLES}}
        return {"nodes": [{"id": node_id, "name": node["name"], "cameras": node["cameras"],
                           "online": now - node["seen"] < 8, "error": node.get("error", "")}
                          for node_id, node in self.nodes.items()],
                "assignments": self.assignments,
                "cameras": {role: {**camera_stats(source), "online": bool(self.frames.get(source) and now - self.frames[source].capture < 0.7),
                                   "age": now - self.frames[source].capture if source in self.frames else None,
                                   "sync_ms": round(self.frames[source].uncertainty * 1000, 1) if source in self.frames else None,
                                   "prediction": self.predictions.get(role), "error": self.errors.get(role),
                                   "reading": self.reading_status(role, now)}
                            for role, source in self.assignments.items()},
                "live": self.live, "attention": self.attention_live,
                "error": self.last_error, "calibration": calibration,
                "displays": self.display_status(now),
                "screen_gaze": {"calibration": self.screen_gaze.cue(now),
                                "participant": self.screen_gaze.participant,
                                "available": list(self.screen_gaze.models)},
                "calibration_participant": self.calibration_participant, "fit_report": self.fit_report,
                "check_report": self.check_report,
                "recording": {"id": self.current["id"], "elapsed": now - self.current["origin"]} if self.current else None}
