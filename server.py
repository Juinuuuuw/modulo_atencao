"""Servidor LAN do experimento. Execute com Python 3.10/3.11."""
from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import secrets
import socket
import sqlite3
import threading
import time
import webbrowser
import zipfile

import cv2
import numpy as np
from flask import Flask, jsonify, request, session, send_file, send_from_directory
from werkzeug.exceptions import HTTPException

from modules.multicamera import Experiment, read_lines, write_json
from modules.target_classifier import ROLES
from modules.screen_gaze import SURFACES, model_key
from modules.storage import AnalysisStore
from modules.validation import (compare_annotators, evaluate_intervals,
                                evaluate_temporal_events, validate_annotations)

ROOT = Path(__file__).resolve().parent


def create_app(output=None, access_key=None, start_worker=True, extractor_factory=None):
    local = ROOT / ".local"
    local.mkdir(exist_ok=True)
    key_file = local / "access-key.txt"
    if access_key is None:
        if not key_file.exists():
            key_file.write_text(secrets.token_urlsafe(18), encoding="utf-8")
        access_key = key_file.read_text(encoding="utf-8").strip()
    app = Flask(__name__, static_folder="web", static_url_path="/static")
    app.config.update(SECRET_KEY=hashlib.sha256(access_key.encode()).hexdigest(),
                      SESSION_COOKIE_NAME="observa_" + hashlib.sha256(access_key.encode()).hexdigest()[:10],
                      MAX_CONTENT_LENGTH=3 * 1024 * 1024, SESSION_COOKIE_HTTPONLY=True,
                      SESSION_COOKIE_SAMESITE="Strict")
    experiment = Experiment(output or ROOT / "sessions", extractor_factory)
    app.extensions["experiment"] = experiment
    app.extensions["access_key"] = access_key
    # Recupera sessões interrompidas sem fingir que foram finalizadas normalmente.
    for path in experiment.output.glob("*/session.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("status") == "recording":
            rows = read_lines(path.parent / "frames.jsonl") + read_lines(path.parent / "predictions.jsonl")
            metadata.update(status="interrupted", duration=max((r["t"] for r in rows), default=0))
            write_json(path, metadata)
    analysis = AnalysisStore(experiment.output)
    try:
        analysis.sync_all()
    except (OSError, ValueError, sqlite3.DatabaseError):
        app.logger.exception("Não foi possível atualizar o índice analítico na inicialização")
    app.extensions["analysis_store"] = analysis

    def sync_analysis(session_id):
        # O índice é secundário: uma falha nele nunca invalida a gravação bruta.
        try:
            analysis.sync_session(session_id)
        except (OSError, ValueError, sqlite3.DatabaseError):
            app.logger.exception("Não foi possível indexar a sessão %s", session_id)
    if start_worker:
        experiment.start_worker()
        atexit.register(experiment.close)

    @app.before_request
    def authorize():
        if not request.path.startswith("/api/"):
            return None
        origin = request.headers.get("Origin")
        if origin and origin != request.host_url.rstrip("/"):
            return jsonify(error="Origem não autorizada"), 403
        if request.path in ("/api/login", "/api/health"):
            return None
        bearer = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if not session.get("authorized") and not secrets.compare_digest(bearer, access_key):
            return jsonify(error="Informe o código de acesso do servidor"), 401

    @app.after_request
    def headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; frame-ancestors 'none'"
        return response

    @app.errorhandler(Exception)
    def errors(error):
        if isinstance(error, HTTPException):
            return jsonify(error=error.description), error.code
        if isinstance(error, (ValueError, KeyError, TypeError, OverflowError)):
            return jsonify(error=str(error)), 400
        app.logger.exception("Falha na requisição")
        return jsonify(error="Falha no servidor. Consulte o log."), 500

    def body():
        value = request.get_json()
        if not isinstance(value, dict):
            raise ValueError("Envie um objeto JSON")
        return value

    def session_path(session_id):
        if not re.fullmatch(r"[a-f0-9]{32}", session_id):
            raise ValueError("Identificador de sessão inválido")
        path = experiment.output / session_id
        if not (path / "session.json").exists():
            raise ValueError("Sessão não encontrada")
        return path

    @app.get("/")
    def index():
        return send_from_directory(ROOT / "web", "index.html")

    @app.get("/alvo")
    def target_display():
        return send_from_directory(ROOT / "web", "target.html")

    @app.get("/api/calibration/cue")
    def calibration_cue():
        with experiment.lock:
            return jsonify(experiment.cue())

    @app.post("/api/calibration/display")
    def calibration_display():
        with experiment.lock:
            experiment.report_display(body())
        return jsonify(ok=True)

    def participant_arg(value):
        value = str(value or "").strip()
        if len(value) > 80:
            raise ValueError("Participante com até 80 caracteres")
        return value or None

    @app.get("/api/screen-gaze")
    def screen_gaze_status():
        display = request.args.get("display", "notebook")
        surface = request.args.get("surface", display)
        if display not in ("notebook", "tv") or surface not in SURFACES:
            raise ValueError("Tela inválida")
        participant = participant_arg(request.args.get("participant"))
        with experiment.lock:
            gaze = experiment.screen_gaze
            cal = gaze.calibration
            if gaze.active and cal["screen"] == display:
                role, surface, participant = cal["role"], cal["surface"], cal["participant"]
            else:
                role = request.args.get("role", "notebook")
            if role not in ROLES:
                raise ValueError("Câmera inválida")
            key = model_key(surface, role)
            model = gaze.models_for(participant).get(key)
            now = time.perf_counter()
            return jsonify(participant=participant or gaze.participant or experiment.calibration_participant,
                           participants=gaze.participants(),
                           calibrated=sorted(gaze.models_for(participant)),
                           library={k: gaze.summary(k, participant) for k in sorted(gaze.models_for(participant))},
                           reading={"role": role, **experiment.reading_status(role, spatial=True)},
                           check_gaze=gaze.checking_position(display, now, experiment.observations),
                           calibration=gaze.cue(),
                           model={"ready": bool(model), "role": role, "surface": surface,
                                  "validation": model["validation"] if model else None,
                                  "validation_text": gaze.validation_text(model["validation"]) if model else None,
                                  "summary": gaze.summary(key, participant)},
                           gaze=gaze.position(key, now, experiment.observations, experiment.camera_labels(),
                                              experiment.displays, participant=participant))

    @app.post("/api/screen-gaze")
    def screen_gaze_action():
        data = body()
        display = str(data.get("display", "notebook"))
        surface = str(data.get("surface") or display)
        role = str(data.get("role", "notebook"))
        with experiment.lock:
            if data["action"] == "cancel":
                experiment.screen_gaze.cancel(display)
            elif data["action"] == "check":
                if experiment.current:
                    raise ValueError("Encerre a gravação antes de medir precisão")
                if role not in ROLES or surface not in SURFACES or display not in ("notebook", "tv"):
                    raise ValueError("Tela ou câmera inválida")
                experiment.screen_gaze.begin_check(surface, role, participant_arg(data.get("participant")), screen=display)
            elif data["action"] == "start":
                if experiment.current:
                    raise ValueError("Encerre a gravação antes de calibrar posição")
                if experiment.calibration and experiment.calibration["started"] is not None:
                    raise ValueError("Aguarde a coleta atual terminar")
                participant = participant_arg(data.get("participant"))
                if not participant or role not in ROLES or display not in ("notebook", "tv") or surface not in SURFACES:
                    raise ValueError("Informe participante, tela e câmera válidos")
                source = experiment.camera_label(role)
                if not source:
                    raise ValueError("Selecione a câmera no painel antes de calibrar posição")
                if experiment.calibration_participant != participant:
                    experiment.cancel_calibration()
                    experiment.calibration_participant = participant
                if experiment.calibration and not experiment.calibration["complete"]:
                    experiment.pause_calibration(True)
                # Por padrão a validação vem logo depois da calibração.
                experiment.screen_gaze.start(surface, role, participant, source,
                                             include_check=data.get("check", True) is not False, screen=display)
            else:
                raise ValueError("Ação inválida")
        return jsonify(ok=True)

    @app.get("/api/screen-gaze/diagnostic")
    def screen_gaze_diagnostic():
        with experiment.lock:
            return jsonify(experiment.screen_gaze.diagnostic())

    @app.get("/api/gaze/live")
    def gaze_live():
        # Visão do operador durante o estudo: não exige a página de alvo aberta.
        participant = participant_arg(request.args.get("participant"))
        with experiment.lock:
            return jsonify(experiment.live_gaze(participant=participant))

    @app.get("/api/health")
    def health():
        return jsonify(ok=True)

    @app.post("/api/login")
    def login():
        if not secrets.compare_digest(str(body().get("key", "")), access_key):
            return jsonify(error="Código incorreto"), 401
        session["authorized"] = True
        return jsonify(ok=True)

    @app.post("/api/logout")
    def logout():
        session.clear()
        return jsonify(ok=True)

    @app.get("/api/connection")
    def connection():
        return jsonify(key=access_key, url=request.host_url.rstrip("/"))

    @app.get("/api/time")
    def server_time():
        return jsonify(monotonic=time.perf_counter(), unix=time.time())

    @app.get("/api/state")
    def state():
        with experiment.lock:
            return jsonify(experiment.state())

    @app.post("/api/nodes/register")
    def register():
        data = body()
        node_id, name, cameras = data["id"], data["name"], data["cameras"]
        if not isinstance(node_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", node_id):
            raise ValueError("ID do cliente inválido")
        if not isinstance(name, str) or not name.strip() or len(name) > 80:
            raise ValueError("Nome inválido")
        if not isinstance(cameras, list) or not 1 <= len(cameras) <= 12:
            raise ValueError("Registre de 1 a 12 câmeras")
        cleaned = []
        for camera in cameras:
            camera_id = str(camera["id"])
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,40}", camera_id):
                raise ValueError("ID de câmera inválido")
            cleaned.append({"id": camera_id, "name": str(camera["name"])[:100],
                            "kind": "nao" if camera.get("kind") == "nao" else "usb"})
        if len({c["id"] for c in cleaned}) != len(cleaned):
            raise ValueError("Câmeras duplicadas")
        with experiment.lock:
            existing = experiment.nodes.get(node_id)
            if existing and existing["cameras"] != cleaned and experiment.current:
                raise ValueError("Encerre a sessão antes de alterar os dispositivos")
            experiment.nodes[node_id] = {"name": name, "cameras": cleaned, "seen": time.perf_counter()}
            if not experiment.current:
                experiment.restore_assignments(node_id)
        return jsonify(ok=True)

    @app.post("/api/nodes/<node_id>/heartbeat")
    def heartbeat(node_id):
        data = body()
        with experiment.lock:
            node = experiment.nodes[node_id]
            node.update(seen=time.perf_counter(), error=str(data.get("error", ""))[:300])
            cameras = [c["id"] for c in node["cameras"]
                       if f"{node_id}:{c['id']}" in experiment.assignments.values()]
        return jsonify(cameras=cameras)

    @app.post("/api/configure")
    def configure():
        with experiment.lock:
            mapping = body()["assignments"]
            for node_id, node in experiment.nodes.items():
                if any(c["kind"] == "nao" for c in node["cameras"]):
                    if sum(bool(s and s.startswith(node_id + ":")) for s in mapping.values()) > 1:
                        raise ValueError("Selecione apenas uma câmera interna por robô NAO")
            experiment.configure(mapping)
        return jsonify(ok=True)

    @app.post("/api/nodes/<node_id>/frame/<camera_id>")
    def ingest(node_id, camera_id):
        with experiment.lock:
            node = experiment.nodes[node_id]
            if camera_id not in [c["id"] for c in node["cameras"]]:
                raise ValueError("Câmera não registrada")
            node["seen"] = time.perf_counter()
        raw = request.get_data()
        if request.mimetype == "application/x-rgb":
            width, height = int(request.headers["X-Width"]), int(request.headers["X-Height"])
            if not 1 <= width <= 1280 or not 1 <= height <= 960 or len(raw) != width * height * 3:
                raise ValueError("Quadro RGB inválido")
            image = cv2.cvtColor(np.frombuffer(raw, np.uint8).reshape(height, width, 3), cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                raise ValueError("Não foi possível codificar o quadro")
            jpeg = encoded.tobytes()
        elif request.mimetype == "image/jpeg":
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[0] > 1080 or image.shape[1] > 1920:
                raise ValueError("JPEG inválido ou maior que 1920×1080")
            jpeg = raw
        else:
            raise ValueError("Formato de imagem não suportado")
        pose = json.loads(request.headers["X-Head-Pose"]) if request.headers.get("X-Head-Pose") else None
        if pose is not None and (not isinstance(pose, list) or len(pose) != 2
                                 or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in pose)):
            raise ValueError("Pose inválida")
        device_time = float(request.headers["X-Device-Time"]) if request.headers.get("X-Device-Time") else None
        if device_time is not None and not math.isfinite(device_time):
            raise ValueError("Timestamp do dispositivo inválido")
        accepted = experiment.receive(f"{node_id}:{camera_id}", image, jpeg,
                                      float(request.headers["X-Capture-Time"]),
                                      float(request.headers["X-Sync-Uncertainty"]),
                                      int(request.headers["X-Sequence"]), pose, device_time)
        return jsonify(accepted=accepted)

    @app.get("/api/preview/<role>")
    def preview(role):
        if role not in ROLES:
            raise ValueError("Papel inválido")
        with experiment.lock:
            frame = experiment.frames.get(experiment.assignments[role])
            if not frame or time.perf_counter() - frame.capture > 2:
                return "", 204
            after = request.args.get("after", type=int)
            if after is not None and frame.sequence == after:
                return "", 204
            jpeg = frame.jpeg
            sequence = frame.sequence
            overlay = experiment.eye_overlay(role, frame)
        response = send_file(io.BytesIO(jpeg), mimetype="image/jpeg")
        response.headers["X-Frame-Sequence"] = str(sequence)
        if overlay is not None:
            response.headers["X-Eye-Overlay"] = json.dumps(overlay, ensure_ascii=True, separators=(",", ":"))
        return response

    @app.post("/api/calibration")
    def calibration():
        data = body()
        with experiment.lock:
            if data["action"] == "fit":
                experiment.begin_calibration(participant=str(data.get("participant", "")).strip(),
                                             profile=str(data.get("profile", "standard")),
                                             automatic=data.get("automatic") is True, chain_check=data.get("chain_check") is True)
            elif data["action"] == "check":
                experiment.begin_calibration(check=True, participant=str(data.get("participant", "")).strip(),
                                             profile=str(data.get("profile", "standard")))
            elif data["action"] == "capture":
                index = data.get("index")
                experiment.capture_step(int(index) if index is not None else None)
            elif data["action"] == "finish":
                experiment.finish_calibration()
            elif data["action"] in ("pause", "resume"):
                experiment.pause_calibration(data["action"] == "pause")
            elif data["action"] == "cancel":
                experiment.cancel_calibration()
            else:
                raise ValueError("Ação inválida")
        return jsonify(ok=True)

    @app.post("/api/recording/start")
    def start():
        data = body()
        with experiment.lock:
            session_id = experiment.start_recording(str(data.get("participant", "")).strip(), str(data.get("override", "")))
        sync_analysis(session_id)
        return jsonify(id=session_id)

    @app.post("/api/recording/stop")
    def stop():
        with experiment.lock:
            session_id = experiment.stop_recording()
        sync_analysis(session_id)
        return jsonify(id=session_id)

    @app.post("/api/events")
    def event():
        data = body()
        with experiment.lock:
            experiment.event(str(data["kind"]), str(data.get("note", "")))
        return jsonify(ok=True)

    @app.get("/api/sessions")
    def sessions():
        result = []
        with experiment.lock:
            for path in sorted(experiment.output.glob("*/session.json"), reverse=True):
                meta = json.loads(path.read_text(encoding="utf-8"))
                result.append({k: meta[k] for k in ("id", "participant", "started_utc", "duration", "status")})
        return jsonify(sorted(result, key=lambda row: row["started_utc"], reverse=True))

    @app.get("/api/sessions/<session_id>")
    def details(session_id):
        with experiment.lock:
            path = session_path(session_id)
            meta = json.loads((path / "session.json").read_text(encoding="utf-8"))
            annotation_metadata_path = path / "annotation_metadata.json"
            annotation_metadata = (json.loads(annotation_metadata_path.read_text(encoding="utf-8"))
                                   if annotation_metadata_path.exists() else {})
            return jsonify(metadata=meta, frames=read_lines(path / "frames.jsonl"),
                           events=read_lines(path / "events.jsonl"),
                           annotations=json.loads((path / "annotations.json").read_text(encoding="utf-8")),
                           annotation_metadata=annotation_metadata)

    @app.get("/api/sessions/<session_id>/frame/<role>/<int:sequence>")
    def recorded_frame(session_id, role, sequence):
        if role not in ROLES or sequence < 0:
            raise ValueError("Quadro inválido")
        return send_from_directory(session_path(session_id) / role, f"{sequence:012d}.jpg")

    @app.post("/api/sessions/<session_id>/annotations")
    def annotate(session_id):
        data = body()
        annotator = str(data.get("annotator", "")).strip()
        if not annotator or len(annotator) > 80:
            raise ValueError("Informe um identificador de avaliador")
        with experiment.lock:
            path = session_path(session_id)
            meta = json.loads((path / "session.json").read_text(encoding="utf-8"))
            if meta["status"] == "recording":
                raise ValueError("Encerre a gravação antes de anotar")
            annotations = json.loads((path / "annotations.json").read_text(encoding="utf-8"))
            annotations[annotator] = validate_annotations(data["intervals"], meta["duration"])
            write_json(path / "annotations.json", annotations)
        sync_analysis(session_id)
        return jsonify(ok=True)

    @app.get("/api/sessions/<session_id>/report")
    def report(session_id):
        with experiment.lock:
            path = session_path(session_id)
            meta = json.loads((path / "session.json").read_text(encoding="utf-8"))
            if meta["status"] == "recording":
                raise ValueError("Encerre a sessão antes de calcular métricas")
            annotations = json.loads((path / "annotations.json").read_text(encoding="utf-8"))
            annotator = request.args.get("annotator", "")
            if annotator not in annotations:
                raise ValueError("Salve as anotações desse avaliador primeiro")
            value = evaluate_intervals(read_lines(path / "predictions.jsonl"), annotations[annotator], meta["duration"])
            value["temporal_iou"] = evaluate_temporal_events(
                read_lines(path / "attention_events.jsonl"), annotations[annotator], meta["duration"])
            value.update(annotator=annotator, session_id=session_id)
            reports_path = path / "reports.json"
            reports = (json.loads(reports_path.read_text(encoding="utf-8"))
                       if reports_path.exists() else {})
            reports[annotator] = value
            write_json(reports_path, reports)
            write_json(path / "report.json", value)  # compatibilidade com sessoes antigas
        sync_analysis(session_id)
        return jsonify(value)

    @app.get("/api/sessions/<session_id>/agreement")
    def agreement(session_id):
        with experiment.lock:
            path = session_path(session_id)
            meta = json.loads((path / "session.json").read_text(encoding="utf-8"))
            annotations = json.loads((path / "annotations.json").read_text(encoding="utf-8"))
            first, second = request.args.get("first", ""), request.args.get("second", "")
            if first not in annotations or second not in annotations:
                raise ValueError("Selecione dois avaliadores com anotacoes salvas")
            value = compare_annotators(annotations[first], annotations[second], meta["duration"],
                                       first_name=first, second_name=second)
            value.update(session_id=session_id)
        return jsonify(value)

    @app.post("/api/sessions/<session_id>/consensus")
    def save_consensus(session_id):
        data = body()
        first, second = str(data.get("first", "")).strip(), str(data.get("second", "")).strip()
        annotator = str(data.get("annotator", "")).strip()
        if not annotator or len(annotator) > 80 or annotator in (first, second):
            raise ValueError("Informe um identificador exclusivo para o consenso")
        with experiment.lock:
            path = session_path(session_id)
            meta = json.loads((path / "session.json").read_text(encoding="utf-8"))
            annotations = json.loads((path / "annotations.json").read_text(encoding="utf-8"))
            if first not in annotations or second not in annotations or first == second:
                raise ValueError("Selecione dois avaliadores diferentes com anotacoes salvas")
            intervals = validate_annotations(data.get("intervals", []), meta["duration"])
            comparison = compare_annotators(annotations[first], annotations[second], meta["duration"],
                                            first_name=first, second_name=second)
            expected = comparison["consensus_draft"]
            expected_seconds = sum(row["end"] - row["start"] for row in expected)
            saved_seconds = sum(row["end"] - row["start"] for row in intervals)
            covered_seconds = sum(max(0.0, min(row["end"], reference["end"])
                                      - max(row["start"], reference["start"]))
                                  for row in intervals for reference in expected)
            if (abs(saved_seconds - expected_seconds) > 0.001
                    or abs(covered_seconds - expected_seconds) > 0.001):
                raise ValueError("O consenso deve cobrir exatamente o tempo anotado pelos dois avaliadores")
            annotations[annotator] = intervals
            write_json(path / "annotations.json", annotations)
            metadata_path = path / "annotation_metadata.json"
            metadata = (json.loads(metadata_path.read_text(encoding="utf-8"))
                        if metadata_path.exists() else {})
            metadata[annotator] = {"kind": "consensus", "sources": [first, second]}
            write_json(metadata_path, metadata)
        sync_analysis(session_id)
        return jsonify(ok=True, annotator=annotator)

    @app.get("/api/dashboard")
    def dashboard():
        requested = request.args.get("session_id") or None
        if requested is not None:
            session_path(requested)
        return jsonify(analysis.dashboard(requested))

    @app.get("/api/dashboard/summary.csv")
    def dashboard_csv():
        value = analysis.dashboard()
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow(["session_id", "participant", "started_utc", "duration_seconds",
                         "status", "prediction_samples", "frames"])
        for row in value["sessions"]:
            writer.writerow([row["id"], row["participant"], row["started_utc"], row["duration"],
                             row["status"], row["predictions"], row["frames"]])
        return send_file(io.BytesIO(buffer.getvalue().encode("utf-8-sig")), mimetype="text/csv",
                         as_attachment=True, download_name="observa-resumo-sessoes.csv")

    @app.get("/api/sessions/<session_id>/predictions.csv")
    def predictions_csv(session_id):
        rows = read_lines(session_path(session_id) / "predictions.jsonl")
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow(["t_seconds", "target", "score", "reason", "attention_state",
                         "ear", "ear_threshold", "gaze_activity", "notebook", "nao", "validation",
                         "tracking_valid_json", "screen_gaze_json"])
        for row in rows:
            attention = row.get("attention") or {}
            cameras = row.get("cameras") or {}
            writer.writerow([row["t"], row["target"], row["score"], row["reason"],
                             attention.get("state"), attention.get("ear"),
                             attention.get("ear_threshold"), attention.get("gaze_activity")] +
                            [cameras.get(role, {}).get("target") for role in ROLES] +
                            [json.dumps({role: bool(cameras.get(role, {}).get("tracking_valid"))
                                         for role in ROLES}, ensure_ascii=False, separators=(",", ":"))] +
                            [json.dumps(row.get("screen_gaze"), ensure_ascii=False, separators=(",", ":"))])
        return send_file(io.BytesIO(buffer.getvalue().encode("utf-8-sig")), mimetype="text/csv",
                         as_attachment=True, download_name=f"{session_id}-predictions.csv")

    @app.get("/api/client.zip")
    def client_zip():
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in ("camera_client.py", "nao_client.py", "iniciar_nao.sh", "requirements-client.txt", "iniciar_cliente.bat", "CLIENTE.md"):
                archive.write(ROOT / name, name)
        buffer.seek(0)
        return send_file(buffer, mimetype="application/zip", as_attachment=True, download_name="cliente-cameras.zip")

    return app


def main():
    parser = argparse.ArgumentParser(description="Servidor do monitor multicâmera na rede local")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, default=ROOT / "sessions")
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--no-local-cameras", action="store_true", help="Não procurar webcams no servidor")
    args = parser.parse_args()
    app = create_app(args.output)
    key = app.extensions["access_key"]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 1))
            lan_ip = sock.getsockname()[0]
    except OSError:
        lan_ip = socket.gethostbyname(socket.gethostname())
    print(f"Painel local: http://localhost:{args.port}", flush=True)
    print(f"Painel na rede: http://{lan_ip}:{args.port}", flush=True)
    print(f"Código de acesso: {key}", flush=True)
    print("Use apenas na rede local do experimento. HTTP não cifra as imagens em trânsito.", flush=True)
    local_stop = threading.Event()
    local_thread = None
    if not args.no_local_cameras:
        from camera_client import run
        client_args = argparse.Namespace(server=f"http://127.0.0.1:{args.port}", name="Servidor / " + socket.gethostname(),
                                         cameras=None, scan=4, backend="auto", width=1280, height=720, fps=30, list=False)
        def local_client():
            try:
                run(client_args, key, local_stop)
            except (SystemExit, Exception) as error:
                print(f"Câmeras locais: {error}", flush=True)
        local_thread = threading.Thread(target=local_client, name="local-cameras", daemon=True)
        local_thread.start()
    if args.open:
        threading.Timer(1.5, webbrowser.open, args=[f"http://localhost:{args.port}/#access={key}"]).start()
    from waitress import serve
    try:
        serve(app, host=args.host, port=args.port, threads=12)
    finally:
        local_stop.set()
        if local_thread:
            local_thread.join(timeout=8)
        app.extensions["experiment"].close()


if __name__ == "__main__":
    main()
