"""Captura webcams do computador cliente e envia somente as selecionadas no painel."""
from __future__ import annotations

import argparse
import getpass
import os
import socket
import threading
import time
import uuid
from urllib.parse import urlsplit

import cv2
import requests


class Clock:
    def __init__(self):
        self.lock = threading.Lock()
        self.offset = None
        self.uncertainty = 0.0

    def sync(self, http, server):
        estimates = []
        for _ in range(3):
            before = time.perf_counter()
            response = http.get(server + "/api/time", timeout=5)
            response.raise_for_status()
            remote = response.json()["monotonic"]
            after = time.perf_counter()
            estimates.append((after - before, remote - (before + after) / 2))
        rtt, offset = min(estimates)
        with self.lock:
            self.offset, self.uncertainty = offset, rtt / 2

    def timestamp(self, local):
        with self.lock:
            if self.offset is None:
                raise ValueError("Relógio ainda não sincronizado")
            return local + self.offset, self.uncertainty


def backend(name):
    if name == "dshow" or (name == "auto" and os.name == "nt"):
        return cv2.CAP_DSHOW
    if name == "msmf":
        return cv2.CAP_MSMF
    return cv2.CAP_ANY


def discover(indices, capture_backend):
    cameras = []
    for index in indices:
        cap = cv2.VideoCapture(index, capture_backend)
        try:
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    cameras.append({"id": str(index), "name": f"Webcam {index}", "kind": "usb"})
        finally:
            cap.release()
    return cameras


class CameraWorker(threading.Thread):
    def __init__(self, args, camera_id, clock, node_id, key):
        super().__init__(daemon=True, name=f"camera-{camera_id}")
        self.args, self.camera_id, self.clock, self.node_id = args, camera_id, clock, node_id
        self.http = requests.Session()
        self.http.trust_env = False
        self.http.headers["Authorization"] = "Bearer " + key
        self.stop_event = threading.Event()
        self.error = ""
        self.frame_lock = threading.Lock()
        self.latest = None

    def open_capture(self):
        # No Windows, o DirectShow cai para ~8 FPS em 1280×720 em várias webcams;
        # o Media Foundation (usado pelo módulo isolado) mantém ~20–30 FPS.
        # A busca de câmeras continua no DirectShow, que é mais rápida.
        if self.args.backend == "auto" and os.name == "nt":
            candidates = (cv2.CAP_MSMF, cv2.CAP_DSHOW)
        else:
            candidates = (backend(self.args.backend),)
        # Algumas câmeras (ex.: infravermelha do Windows Hello, webcams simples) não
        # aceitam 1280×720: tenta depois sem forçar resolução, como na busca inicial.
        attempts = []
        for forced in (True, False):
            for api in candidates:
                cap = cv2.VideoCapture(int(self.camera_id), api)
                if forced:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if cap.isOpened() and cap.read()[0]:
                    if not forced:
                        print(f"Webcam {self.camera_id}: usando a resolução padrão da câmera "
                              f"({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}).", flush=True)
                    return cap
                attempts.append(f"{cap.getBackendName() if cap.isOpened() else api}"
                                f"{' ' + str(self.args.width) + '×' + str(self.args.height) if forced else ' padrão'}")
                cap.release()
        self.open_attempts = attempts
        return None

    def capture(self):
        cap = None
        sequence = 0
        try:
            while not self.stop_event.is_set():
                if cap is None:
                    cap = self.open_capture()
                    if cap is None:
                        self.error = (f"Webcam {self.camera_id}: sem imagem (ocupada, desconectada ou câmera "
                                      f"infravermelha). Tentativas: {', '.join(getattr(self, 'open_attempts', []))}")
                        self.stop_event.wait(2)
                        continue
                ok, frame = cap.read()
                captured = time.perf_counter()
                if not ok:
                    self.error = f"Webcam {self.camera_id}: sem imagem (ocupada ou desconectada)"
                    cap.release()
                    cap = None
                    self.stop_event.wait(2)
                    continue
                sequence += 1
                with self.frame_lock:
                    self.latest = (sequence, captured, frame)
        finally:
            if cap:
                cap.release()

    def run(self):
        # A leitura não espera o envio HTTP: mantém somente o quadro mais novo.
        # Evita acumular quadros antigos quando a rede ou o servidor está lento.
        reader = threading.Thread(target=self.capture, daemon=True)
        reader.start()
        sent = 0
        last_new = time.perf_counter()
        try:
            while not self.stop_event.is_set():
                started = time.perf_counter()
                with self.frame_lock:
                    latest = self.latest
                if latest is None or latest[0] == sent:
                    # A abertura/leitura do driver pode travar sem erro (comum ao reabrir no Windows).
                    stalled = started - last_new
                    if stalled > 8 and (not self.error or "sem quadros" in self.error):
                        self.error = (f"Webcam {self.camera_id}: sem quadros há {stalled:.0f} s (o driver da câmera pode "
                                      f"ter travado). Feche este conector e abra de novo.")
                    self.stop_event.wait(0.02)
                    continue
                last_new = started
                sequence, captured, frame = latest
                sent = sequence
                try:
                    # Limita dimensões mesmo quando o driver ignora o pedido.
                    height, width = frame.shape[:2]
                    scale = min(1, self.args.width / width, self.args.height / height)
                    if scale < 1:
                        frame = cv2.resize(frame, (round(width * scale), round(height * scale)))
                    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if not ok:
                        raise ValueError("Falha ao codificar JPEG")
                    timestamp, uncertainty = self.clock.timestamp(captured)
                    response = self.http.post(
                        f"{self.args.server}/api/nodes/{self.node_id}/frame/{self.camera_id}",
                        data=encoded.tobytes(), timeout=4,
                        headers={"Content-Type": "image/jpeg", "X-Capture-Time": str(timestamp),
                                 "X-Sync-Uncertainty": str(uncertainty), "X-Sequence": str(sequence)})
                    response.raise_for_status()
                    self.error = ""
                except (requests.RequestException, ValueError) as error:
                    self.error = f"Webcam {self.camera_id}: {error}"
                self.stop_event.wait(max(0, 1 / self.args.fps - (time.perf_counter() - started)))
        finally:
            self.stop_event.set()
            reader.join(timeout=3)
            self.http.close()


def main():
    parser = argparse.ArgumentParser(description="Conector de webcams para o servidor LAN")
    parser.add_argument("--server", help="Ex.: http://192.168.1.10:8765")
    parser.add_argument("--name", default=socket.gethostname())
    parser.add_argument("--cameras", help="Índices separados por vírgula; omitir para procurar")
    parser.add_argument("--scan", type=int, default=4, help="Quantidade de índices a procurar (0..N-1)")
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf"), default="auto")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--list", action="store_true", help="Só listar as webcams disponíveis")
    args = parser.parse_args()
    if not (1 <= args.fps <= 30 and 1 <= args.width <= 1920 and 1 <= args.height <= 1080 and 1 <= args.scan <= 12):
        parser.error("Use 1–30 FPS, até 1920×1080 e busca de 1–12 índices")
    run(args)


def normalize_server(value, default_port=8765):
    """Aceita só o IP/nome (completa http:// e a porta padrão) ou o endereço completo."""
    value = value.strip()
    if not value:
        raise ValueError("Informe o endereço do servidor, por exemplo 192.168.16.162")
    if "://" not in value:
        value = "http://" + value
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("Informe um endereço como 192.168.16.162 ou http://192.168.16.162:8765")
    netloc = parts.netloc if parts.port is not None or parts.scheme == "https" else f"{parts.netloc}:{default_port}"
    return f"{parts.scheme}://{netloc}{parts.path.rstrip('/')}"


def run(args, key=None, shutdown_event=None):
    shutdown_event = shutdown_event or threading.Event()
    indices = [int(x) for x in args.cameras.split(",")] if args.cameras else range(args.scan)
    print("Procurando webcams. Feche outros programas que estejam usando a câmera.", flush=True)
    cameras = discover(indices, backend(args.backend))
    for camera in cameras:
        print(f"Disponível: {camera['name']}", flush=True)
    if not cameras:
        raise SystemExit("Nenhuma webcam disponível. Confira conexão/permissões e tente --backend msmf.")
    if args.list:
        return
    args.server = normalize_server(args.server or input("Endereço do servidor (ex.: 192.168.16.162): "))
    key = key or os.environ.get("MONITOR_ACCESS_KEY") or getpass.getpass("Código de acesso do painel: ")
    http = requests.Session()
    http.trust_env = False
    http.headers["Authorization"] = "Bearer " + key
    clock, workers = Clock(), {}
    node_id = uuid.uuid4().hex
    registered, last_sync = False, 0.0
    try:
        while not shutdown_event.is_set():
            try:
                if not registered:
                    response = http.post(args.server + "/api/nodes/register", timeout=5,
                                         json={"id": node_id, "name": args.name, "cameras": cameras})
                    response.raise_for_status()
                    registered = True
                    print("Conectado. Selecione as câmeras no painel do servidor.", flush=True)
                if time.perf_counter() - last_sync > 30:
                    clock.sync(http, args.server)
                    last_sync = time.perf_counter()
                response = http.post(f"{args.server}/api/nodes/{node_id}/heartbeat", timeout=5,
                                     json={"error": "; ".join(w.error for w in workers.values() if w.error)})
                response.raise_for_status()
                desired = set(response.json()["cameras"])
                for camera_id in list(workers):
                    if camera_id not in desired:
                        workers[camera_id].stop_event.set()
                        workers[camera_id].join(timeout=6)
                        del workers[camera_id]
                for camera_id in desired - workers.keys():
                    workers[camera_id] = CameraWorker(args, camera_id, clock, node_id, key)
                    workers[camera_id].start()
            except requests.RequestException as error:
                print(f"Conexão: {error}. Nova tentativa em 3 segundos.", flush=True)
                registered = False
                last_sync = 0
                shutdown_event.wait(2)
            shutdown_event.wait(1)
    except KeyboardInterrupt:
        print("Encerrando câmeras...", flush=True)
    finally:
        for worker in workers.values():
            worker.stop_event.set()
        for worker in workers.values():
            worker.join(timeout=6)
        http.close()


if __name__ == "__main__":
    main()
